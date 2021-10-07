# calculate exemplars of all previous classes based on current network

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
from utils import set
from data import imagenet

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-batch_size',type=int,default=128)
    parser.add_argument('-dataset',type=str,default='imagenet')
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
    parser.add_argument('-num_samples_per_task',type=int,default=20)
    args=parser.parse_args()

    device = torch.device(args.gpu_info if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = imagenet.ImageFolder(traindir,transform_train)


#     memory=np.zeros((100,args.num_samples_per_task,32,32,3),dtype=np.uint8)
#     memory_label=np.zeros((100,args.num_samples_per_task),dtype=np.int64)
#
#
#     if args.dataset=='cifar100':
#         total_num_class=100
#     num_tasks=(total_num_class-args.base_num_classes)//args.incre_num_classes
#
#     set.setup_seed(args.seed)
#     index_train=np.array(range(total_num_class))
#     np.random.shuffle(index_train)
#
#     lable_real_to_logic={}
#     for i in range(index_train.__len__()):
#         lable_real_to_logic[index_train[i]]=i
#
#     net=resnet.resnet32(num_classes=args.base_num_classes)
#     net = net.to(device)
#
#     for task_id in range(num_tasks+1):
#         print('current task is : {}'.format(task_id))
#         if task_id==1:
#             num_old_classes=args.base_num_classes
#         elif task_id>1:
#             num_old_classes+=args.incre_num_classes
#
#         if task_id==0:
#             index_cur=index_train[:args.base_num_classes]
#         else:
#             index_cur=index_train[args.base_num_classes+(task_id-1)*args.incre_num_classes:args.base_num_classes+task_id*args.incre_num_classes]
#         trainset = cifar.CIFAR100(root=args.dataset_dir, train=True, transform=transform_train, index=index_cur, target_transform=lable_real_to_logic)
#
#         if task_id>0:
#             trainset.data=np.array(np.vstack((trainset.data,memory[:num_old_classes,:,:,:,:].reshape([-1,32,32,3]))))
#             trainset.targets=np.array(np.append(trainset.targets,memory_label[:num_old_classes,:].reshape([-1]),axis=0))
#
#
#         if task_id>0:
#             ref_net=copy.deepcopy(net)
#             ref_net=ref_net.to(device)
#             ref_net.eval()
#             in_features=ref_net.fc.in_features
#             out_features=ref_net.fc.out_features
#             net.fc=nn.Linear(in_features,out_features+args.incre_num_classes)
#             net.fc.weight.data[:out_features]=ref_net.fc.weight.data
#             net.fc.bias.data[:out_features]=ref_net.fc.bias.data
#             net=net.to(device)
#
#         trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
#         optimizer = torch.optim.SGD(net.parameters(), lr=args.lr_initial, momentum=0.9, weight_decay=5e-4)
#         lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[80, 120],
#                                                             gamma=args.lr_decay)
#         loss_CE = nn.CrossEntropyLoss().to(device)
#         loss_KL = nn.KLDivLoss().to(device)
#
#         if args.resume_skip_base and task_id==0:
#             net = torch.load('./../checkpoints/icarl/base_{}_incre_{}_task_{}_model'.format(args.base_num_classes,args.incre_num_classes,task_id))
#             net = net.to(device)
#         else:
#             net.train()
#             for epoch in tqdm(range(args.epochs)):
#                 net.train()
#                 train_loss_classification=0
#                 train_loss_kd=0
#                 for (input,label) in trainloader:
#                     optimizer.zero_grad()
#                     input = input.to(device)
#                     label = label.to(device)
#                     output = net(input)
#                     loss = loss_CE(output, label)
#                     train_loss_classification += loss
#
#                     if task_id > 0:
#                         ref_output = ref_net(input)
#                         loss_kd = loss_KL(F.log_softmax(output[:, :num_old_classes] / 2.0, dim=1), F.softmax(ref_output.detach() / 2.0, dim=1)) * 2.0 * 2.0 * 0.25 * num_old_classes
#                         train_loss_kd += args.kd_loss_weight * loss_kd
#                         loss += args.kd_loss_weight * loss_kd
#
#                     loss.backward()
#                     optimizer.step()
#                 lr_scheduler.step()
#                 #print('epoch {} : loss_classification {} loss_kd {} '.format(epoch,train_loss_classification,train_loss_kd))
#
#                 # net.eval()
#                 # with torch.no_grad():
#                 #     total_index = index_train[:args.base_num_classes + task_id * args.incre_num_classes]
#                 #     evalset = cifar.CIFAR100(root=args.dataset_dir, train=False, transform=transform_test,
#                 #                              index=total_index,
#                 #                              target_transform=lable_real_to_logic)
#                 #     evalloader = torch.utils.data.DataLoader(evalset, batch_size=100, shuffle=True, num_workers=2)
#                 #     acc_all = Compute.compute_accuracy(net, evalloader, device)
#                 #     print('after epoch {} : overall average accuracy is {}'.format(epoch, acc_all))
#
#
#         save_path = './../checkpoints/icarl/base_{}_incre_{}_task_{}_model'.format(args.base_num_classes,args.incre_num_classes,task_id)
#         torch.save(net, save_path)
#         args.epochs=60
#
#         net.eval()
#         acc = []
#         with torch.no_grad():
#             for previous_id in range(task_id+1):
#                 if previous_id == 0:
#                     index_pre = index_train[:args.base_num_classes]
#                 else:
#                     index_pre = index_train[args.base_num_classes + (
#                                 previous_id - 1) * args.incre_num_classes:args.base_num_classes + previous_id * args.incre_num_classes]
#                 evalset_pre = cifar.CIFAR100(root=args.dataset_dir, train=False, transform=transform_test, index=index_pre,target_transform=lable_real_to_logic)
#                 evalloader_pre = torch.utils.data.DataLoader(evalset_pre, batch_size=100, shuffle=True, num_workers=2)
#                 acc_pre=Compute.compute_accuracy(net,evalloader_pre,device)
#                 print('task {} : evaluation accuracy : {} '.format(previous_id,acc_pre))
#                 acc.append(acc_pre)
#             total_index=index_train[:args.base_num_classes+task_id*args.incre_num_classes]
#             evalset = cifar.CIFAR100(root=args.dataset_dir, train=False, transform=transform_test, index=total_index,
#                                          target_transform=lable_real_to_logic)
#             evalloader = torch.utils.data.DataLoader(evalset, batch_size=100, shuffle=True, num_workers=2)
#             acc_all = Compute.compute_accuracy(net,evalloader,device)
#             print('after training task {} : overall average accuracy is {}'.format(task_id,acc_all))
#             acc.append(acc_all)
#         print('')
#
#         if task_id==num_tasks:
#             print('after training all tasks, average accuracy is {} '.format(np.mean(acc)))
#
#         # herding
#         herdnet=copy.deepcopy(net)
#         if task_id==0:
#             start=0
#             end=args.base_num_classes
#         else:
#             start=0
#             end=args.base_num_classes+task_id*args.incre_num_classes
#         for i in range(start,end,1):
#             data=cifar.CIFAR100(root=args.dataset_dir, train=True, transform=transform_train, index=[index_train[i]],target_transform=lable_real_to_logic)
#             loader=torch.utils.data.DataLoader(data,batch_size=100, shuffle=False, num_workers=2)
#             feature=Compute_Feature(loader,herdnet,device,data.__len__())
#             mean_feature=np.mean(feature,axis=0)
#             dy_mean=mean_feature
#             feature_inv = feature.T
#             feature_inv = feature_inv / np.linalg.norm(feature_inv, axis=0)
#
#             cnt=0
#             chosen=[]
#             while not(cnt==min(args.num_samples_per_task,500)):
#                 sim=np.dot(dy_mean,feature_inv)
#                 index=np.argmax(sim)
#                 if index not in chosen:
#                     chosen.append(index)
#                     memory[i,cnt,:,:,:]=data.data[index]
#                     memory_label[i,cnt]=index_train[i]
#                     cnt += 1
#                 dy_mean=dy_mean+mean_feature-feature[index]
#
#
#
# if __name__=='__main__':
#     main()