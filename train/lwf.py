import sys
sys.path.append('..')
import data.cifar as cifar
import torchvision.transforms as transforms
import argparse
import numpy as np
import torch
import network.resnet as resnet
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from utils import Compute
import copy
from utils.Compute import Compute_One_Hot
from torch.autograd import Variable

def MultiClassCrossEntropy(logits, labels, T):
	# Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
	labels = Variable(labels.data, requires_grad=False).cuda()
	outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
	labels = torch.softmax(labels/T, dim=1)
	# print('outputs: ', outputs)
	# print('labels: ', labels.shape)
	outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
	outputs = -torch.mean(outputs, dim=0, keepdim=False)
	# print('OUT: ', outputs)
	return Variable(outputs.data, requires_grad=True).cuda()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-batch_size',type=int,default=64)
    parser.add_argument('-dataset',type=str,default='cifar100')
    parser.add_argument('-dataset_dir',type=str,default='./../datasets/cifar')
    parser.add_argument('-seed',type=int,default=1993)
    parser.add_argument('-base_num_classes',type=int,default=10)
    parser.add_argument('-incre_num_classes',type=int,default=10)
    parser.add_argument('-lr_initial',type=float,default=0.1)
    parser.add_argument('-lr_decay',type=float,default=0.1)
    parser.add_argument('-epochs',type=int,default=40)
    parser.add_argument('-gpu_info',type=str,default='cuda:0')
    parser.add_argument('-resume_skip_base',action='store_true')
    parser.add_argument('-kd_loss_weight',type=float,default=1.0)
    args=parser.parse_args()

    device=torch.device(args.gpu_info if torch.cuda.is_available() else "cpu")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023))
    ])

    if args.dataset=='cifar100':
        total_num_class=100
    num_tasks=(total_num_class-args.base_num_classes)//args.incre_num_classes

    np.random.seed(args.seed)
    index_train=np.array(range(total_num_class))
    np.random.shuffle(index_train)

    lable_real_to_logic={}
    for i in range(index_train.__len__()):
        lable_real_to_logic[index_train[i]]=i

    net=resnet.resnet32(num_classes=args.base_num_classes)
    net = net.to(device)

    for task_id in range(num_tasks+1):
        print('current task is : {}'.format(task_id))
        if task_id==1:
            num_old_classes=args.base_num_classes
        elif task_id>1:
            num_old_classes+=args.incre_num_classes

        if task_id==0:
            index_cur=index_train[:args.base_num_classes]
        else:
            index_cur=index_train[args.base_num_classes+(task_id-1)*args.incre_num_classes:args.base_num_classes+task_id*args.incre_num_classes]
        trainset = cifar.CIFAR100(root=args.dataset_dir, train=True, transform=transform_train, index=index_cur, target_transform=lable_real_to_logic)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr_initial, momentum=0.9, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[28,36],gamma=args.lr_decay)

        if task_id>0:
            ref_net=copy.deepcopy(net)
            ref_net=ref_net.to(device)
            ref_net.eval()
            in_features=ref_net.fc.in_features
            out_features=ref_net.fc.out_features
            net.fc=nn.Linear(in_features,out_features+args.incre_num_classes)
            nn.init.kaiming_normal_(net.fc.weight, nonlinearity='sigmoid')
            net.fc.weight.data[:out_features]=ref_net.fc.weight.data
            net.fc.bias.data[:out_features]=ref_net.fc.bias.data
            net=net.to(device)

        if args.resume_skip_base and task_id==0:
            net = torch.load('./../checkpoints/lwf/base_{}_incre_{}_task_{}_model'.format(args.base_num_classes,args.incre_num_classes,task_id))
            net = net.to(device)
        else:
            net.train()
            for epoch in tqdm(range(args.epochs)):
                train_loss=0
                for (input,label) in trainloader:
                    optimizer.zero_grad()
                    input = input.to(device)
                    label = label.to(device)
                    output = net(input)

                    if task_id > 0:
                        ref_output = ref_net(input)
                        ref_output = torch.softmax(ref_output,dim=1)
                        one_hot_label=Compute_One_Hot(label,num_old_classes+args.incre_num_classes,device)
                        one_hot_label[:,:num_old_classes]=ref_output/1.2
                        output[:,:num_old_classes]=output[:,:num_old_classes]/1.2
                        loss=F.binary_cross_entropy_with_logits(output,one_hot_label)
                    else:
                        one_hot_label = Compute_One_Hot(label, args.base_num_classes, device)
                        loss=F.binary_cross_entropy_with_logits(output,one_hot_label)
                    train_loss += loss
                    loss.backward()
                    optimizer.step()
                lr_scheduler.step()
                print('epoch {} : loss {} '.format(epoch,train_loss))


        save_path = './../checkpoints/lwf/base_{}_incre_{}_task_{}_model'.format(args.base_num_classes,args.incre_num_classes,task_id)
        torch.save(net, save_path)

        net.eval()
        with torch.no_grad():
            acc=[]
            for previous_id in range(task_id+1):
                if previous_id == 0:
                    index_pre = index_train[:args.base_num_classes]
                else:
                    index_pre = index_train[args.base_num_classes + (
                                previous_id - 1) * args.incre_num_classes:args.base_num_classes + previous_id * args.incre_num_classes]
                evalset_pre = cifar.CIFAR100(root=args.dataset_dir, train=False, transform=transform_test, index=index_pre,target_transform=lable_real_to_logic)
                evalloader_pre = torch.utils.data.DataLoader(evalset_pre, batch_size=100, shuffle=True, num_workers=2)
                acc_pre=Compute.compute_accuracy(net,evalloader_pre,device)
                print('task {} : evaluation accuracy : {} '.format(previous_id,acc_pre))
                acc.append(acc_pre)
            total_index=index_train[:args.base_num_classes+task_id*args.incre_num_classes]
            evalset = cifar.CIFAR100(root=args.dataset_dir, train=False, transform=transform_test, index=total_index,
                                         target_transform=lable_real_to_logic)
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=100, shuffle=True, num_workers=2)
            acc_all = Compute.compute_accuracy(net,evalloader,device)
            print('after training task {} : overall average accuracy is {}'.format(task_id,acc_all))
        print('')

if __name__=='__main__':
    main()