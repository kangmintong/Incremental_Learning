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
from utils.Compute import Compute_Feature
from PIL import Image
from utils import grad_cam
from utils import set
from utils import image_read_save


class trans(nn.Module):
    def __init__(self, num_classes,*args):
        super(trans, self).__init__()
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 , num_classes)
        self.gradients=[]

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def forward(self, x):
        self.gradients = []
        x.register_hook(self.save_gradient)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

CNT=0

def main():
    global CNT
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
    parser.add_argument('-kd_loss_weight',type=float,default=1.0)
    parser.add_argument('-gradcam_loss_weight', type=float, default=1.)
    parser.add_argument('-num_samples_per_task',type=int,default=20)
    args=parser.parse_args()

    memory=np.zeros((100,args.num_samples_per_task,32,32,3),dtype=np.uint8)
    memory_label=np.zeros((100,args.num_samples_per_task),dtype=np.int64)

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


    set.setup_seed(args.seed)
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

        if task_id>0:
            trainset.data=np.array(np.vstack((trainset.data,memory[:num_old_classes,:,:,:,:].reshape([-1,32,32,3]))))
            trainset.targets=np.array(np.append(trainset.targets,memory_label[:num_old_classes,:].reshape([-1]),axis=0))

        if task_id>0:
            ref_net = copy.deepcopy(net)
            ref_net = ref_net.to(device)
            ref_net.eval()
            in_features = ref_net.fc.in_features
            out_features = ref_net.fc.out_features
            net.fc = nn.Linear(in_features, out_features + args.incre_num_classes)
            net.fc.weight.data[:out_features] = ref_net.fc.weight.data
            net.fc.bias.data[:out_features] = ref_net.fc.bias.data
            net = net.to(device)
            # ref_net=resnet.resnet32(num_classes=num_old_classes)
            # ref_net=ref_net.to(device)
            # model_dict=ref_net.state_dict()
            # ref_net_dict = torch.load('./../checkpoints/icarl/base_{}_incre_{}_task_{}_model'.format(
            #     args.base_num_classes,args.incre_num_classes,task_id-1))
            # ref_net_dict = ref_net_dict.state_dict()
            # model_dict.update(ref_net_dict)
            # ref_net.load_state_dict(ref_net_dict)
            # ref_net.eval()
            #
            # net.fc=nn.Linear(64, num_old_classes+args.incre_num_classes)
            # net_dict = torch.load('./../checkpoints/icarl/base_{}_incre_{}_task_{}_model'.format(
            #     args.base_num_classes,args.incre_num_classes,task_id-1))
            # net_dict = net_dict.state_dict()
            # net_dict = {k: v for k, v in net_dict.items() if 'fc' in k }
            # net.fc.weight.data[:num_old_classes]=net_dict['fc.weight'].clone()
            # net.fc.bias.data[:num_old_classes]=net_dict['fc.bias'].clone()
            # net=net.to(device)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr_initial, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[args.epochs//2, args.epochs*3//4],
                                                            gamma=args.lr_decay)
        loss_CE = nn.CrossEntropyLoss().to(device)
        loss_KL = nn.KLDivLoss().to(device)


        if args.resume_skip_base and task_id==0:
            net = torch.load('./../checkpoints/icarl/base_{}_incre_{}_task_{}_model'.format(args.base_num_classes,args.incre_num_classes,task_id))
            net = net.to(device)
        else:
            net.train()
            for epoch in tqdm(range(args.epochs)):
                net.train()
                train_loss_classification=0
                train_loss_kd=0
                train_loss_gradcam=0

                for (input,label) in trainloader:
                    input = input.to(device)
                    label = label.to(device)
                    output,feature_new,_ = net(input,True)
                    loss = loss_CE(output, label)
                    train_loss_classification += loss

                    if task_id > 0:
                        ref_output,feature_old,_ = ref_net(input,True)
                        loss_kd = loss_KL(F.log_softmax(output[:, :num_old_classes] / 2.0, dim=1), F.softmax(ref_output.detach() / 2.0, dim=1)) * 2.0 * 2.0 * 0.25 * num_old_classes
                        train_loss_kd += args.kd_loss_weight * loss_kd
                        loss += args.kd_loss_weight * loss_kd

                        batch=len(input)
                        index = output[:,:num_old_classes].argmax(dim=-1).view(-1, 1)
                        index=index.requires_grad_(False)
                        onehot_old = torch.zeros_like(ref_output)
                        onehot_old.scatter_(-1, index, 1.)
                        out_o = torch.sum(onehot_old * ref_output)
                        out_o.backward(retain_graph=True)
                        grads_o = ref_net.gradients[-1]

                        onehot_new = torch.zeros_like(output)
                        onehot_new.scatter_(-1, index, 1.)
                        out_n = torch.sum(onehot_new * output)
                        out_n.backward(retain_graph=True)
                        grads_n = net.gradients[-1]

                        weight_o = grads_o.mean(dim=1).view(batch, -1, 1, 1)
                        weight_n = grads_n.mean(dim=1).view(batch, -1, 1, 1)
                        # print(weight_o.shape)
                        # print(feature_old.shape)
                        cam_o = torch.sum(feature_old * weight_o, dim=1).squeeze()
                        #print(cam_o.shape)
                        cam_n = torch.sum(feature_new * weight_n, dim=1).squeeze()

                        # if CNT==0 and epoch>0:
                        #     image_read_save.save_cam(cam_o[0],'ref_attention.jpg')
                        #     image_read_save.save_cam(cam_n[0],'new_attention.jpg')
                        #     CNT+=1

                        cam_o = F.normalize(cam_o.view(batch, -1), p=2, dim=-1)
                        cam_n = F.normalize(cam_n.view(batch, -1), p=2, dim=-1)
                        #loss_gradcam = (cam_o - cam_n).norm(p=1, dim=1).mean()
                        loss_gradcam = F.normalize(cam_n-cam_o, p=2, dim=1).mean()
                        print(loss_gradcam)

                        train_loss_gradcam += args.gradcam_loss_weight * loss_gradcam
                        loss += args.gradcam_loss_weight * loss_gradcam
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # print('after')
                    # print(torch.sum(net.fc.weight.data))
                lr_scheduler.step()
                print('epoch {} : loss_classification {} loss_kd {} loss_gradcam {} '.format(epoch,train_loss_classification,train_loss_kd,train_loss_gradcam))

                # print('before')
                # print(net.state_dict()['fc.bias'][:num_old_classes] - ref_net.state_dict()['fc.bias'])

                net.eval()
                with torch.no_grad():
                    total_index = index_train[:args.base_num_classes + task_id * args.incre_num_classes]
                    evalset = cifar.CIFAR100(root=args.dataset_dir, train=False, transform=transform_test,
                                             index=total_index,
                                             target_transform=lable_real_to_logic)
                    evalloader = torch.utils.data.DataLoader(evalset, batch_size=100, shuffle=True, num_workers=2)
                    acc_all = Compute.compute_accuracy(net, evalloader, device)
                    print('after epoch {} : overall average accuracy is {}'.format(epoch, acc_all))


        save_path = './../checkpoints/icarl/base_{}_incre_{}_task_{}_model'.format(args.base_num_classes,args.incre_num_classes,task_id)
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

        # herding
        herdnet=copy.deepcopy(net)
        if task_id==0:
            start=0
            end=args.base_num_classes
        else:
            start=args.base_num_classes+(task_id-1)*args.incre_num_classes
            end=args.base_num_classes+task_id*args.incre_num_classes
        for i in range(start,end,1):
            data=cifar.CIFAR100(root=args.dataset_dir, train=True, transform=transform_train, index=[index_train[i]],target_transform=lable_real_to_logic)
            loader=torch.utils.data.DataLoader(data,batch_size=100, shuffle=False, num_workers=2)
            feature=Compute_Feature(loader,herdnet,device,data.__len__())
            mean_feature=np.mean(feature,axis=0)
            dy_mean=mean_feature
            feature_inv = feature.T
            feature_inv = feature_inv / np.linalg.norm(feature_inv, axis=0)

            cnt=0
            chosen=[]
            while not(cnt==min(args.num_samples_per_task,500)):
                sim=np.dot(dy_mean,feature_inv)
                index=np.argmax(sim)
                if index not in chosen:
                    chosen.append(index)
                    memory[i,cnt,:,:,:]=data.data[index]
                    memory_label[i,cnt]=index_train[i]
                    cnt += 1
                dy_mean=dy_mean+mean_feature-feature[index]



if __name__=='__main__':
    main()