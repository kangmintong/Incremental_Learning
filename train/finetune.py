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

lable_real_to_logic={}

def transform_label(x):
    return lable_real_to_logic[x]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-batch_size',type=int,default=128)
    parser.add_argument('-dataset',type=str,default='cifar100')
    parser.add_argument('-dataset_dir',type=str,default='./../datasets/cifar')
    parser.add_argument('-seed',type=int,default=1993)
    parser.add_argument('-base_num_classes',type=int,default=50)
    parser.add_argument('-incre_num_classes',type=int,default=10)
    parser.add_argument('-lr_initial',type=float,default=0.1)
    parser.add_argument('-lr_decay',type=float,default=0.1)
    parser.add_argument('-epochs',type=int,default=160)
    parser.add_argument('-gpu_info',type=str,default='cuda:0')
    parser.add_argument('-resume_skip_base',action='store_true')
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

    for i in range(index_train.__len__()):
        lable_real_to_logic[index_train[i]]=i

    net=resnet.resnet32(num_classes=args.base_num_classes)
    net = net.to(device)

    for task_id in range(num_tasks+1):
        print('current task is : {}'.format(task_id))
        if task_id==0:
            index_cur=index_train[:args.base_num_classes]
        else:
            index_cur=index_train[args.base_num_classes+(task_id-1)*args.incre_num_classes:args.base_num_classes+task_id*args.incre_num_classes]
        trainset = cifar.CIFAR100(root=args.dataset_dir, train=True, transform=transform_train, index=index_cur, target_transform=lable_real_to_logic)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr_initial, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[args.epochs/2, args.epochs*3/4],gamma=args.lr_decay)
        loss_CE = nn.CrossEntropyLoss().to(device)

        if task_id>0:
            ref_net=copy.deepcopy(net)
            ref_net=ref_net.to(device)
            in_features=ref_net.fc.in_features
            out_features=ref_net.fc.out_features
            net.fc=nn.Linear(in_features,out_features+args.incre_num_classes)
            net.fc.weight.data[:out_features]=ref_net.fc.weight.data
            net.fc.bias.data[:out_features]=ref_net.fc.bias.data
            net=net.to(device)

            args.epochs=60

        if args.resume_skip_base and task_id==0:
            net = torch.load('./../checkpoints/finetune/task_{}_model'.format(task_id))
            net = net.to(device)
        else:
            net.train()
            for epoch in tqdm(range(args.epochs)):
                lr_scheduler.step()
                for (input,label) in trainloader:
                    optimizer.zero_grad()
                    input=input.to(device)
                    label=label.to(device)
                    output=net(input)
                    loss=loss_CE(output,label)
                    loss.backward()
                    optimizer.step()
                #print(net.state_dict()['fc.weight'])

        save_path = './../checkpoints/finetune/task_{}_model'.format(task_id)
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
            print('after training task {} : overall average accuracy is {}'.format(task_id,np.sum(acc)/(task_id+1)))
        print('')



if __name__=='__main__':
    main()