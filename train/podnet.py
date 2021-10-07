
import sys
sys.path.append('..')
import data.cifar as cifar
import torchvision.transforms as transforms
import argparse
import numpy as np
import torch
import network.resnet_pod as resnet_pod
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from utils import Compute
import copy
from utils import set
from utils.Compute import Compute_Feature_podnet
from utils.Compute import Compute_Pod_Space_Loss
from utils import net_copy
import math

cur_features = []
ref_features = []
old_scores = []
new_scores = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    if len(old_scores)==0:
        old_scores = outputs
    else:
        old_scores = old_scores + outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    if len(new_scores)==0:
        new_scores = outputs
    else:
        new_scores = new_scores + outputs

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
    parser.add_argument('-kd_loss_weight',type=float,default=5.0)
    parser.add_argument('-mr_loss_weight', type=float, default=1.0)
    parser.add_argument('-pod_space_loss_weight', type=float, default=1.0)
    parser.add_argument('-dist', type=float, default=0.5)
    parser.add_argument('-num_samples_per_task',type=int,default=20)
    parser.add_argument('-K',type=int,default=2)
    parser.add_argument('-fc_num',type=int,default=1)
    args=parser.parse_args()

    global ref_features
    global cur_features
    global old_scores
    global new_scores

    set.setup_seed(args.seed)
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

    index_train=np.array(range(total_num_class),dtype=np.uint64)
    np.random.shuffle(index_train)
    lable_real_to_logic={}
    for i in range(index_train.__len__()):
        lable_real_to_logic[index_train[i]]=i

    net=resnet_pod.resnet32(num_classes=args.base_num_classes, fc_num=args.fc_num, device=device)
    net = net.to(device)
    num_old_classes=0

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


        # two parts of fc equal one whole fc ??
        if task_id==1:
            ref_net=copy.deepcopy(net)
            ref_net=ref_net.to(device)
            ref_net.eval()
            in_features=ref_net.multi_fc.in_features
            out_features=ref_net.multi_fc.out_features

            fcs=[]
            fc1s=[]
            for i in range(args.fc_num):
                new_fc=resnet_pod.SplitCosineLinear(in_features,out_features,args.incre_num_classes).to(device)
                new_fc.fc1.weight.data=ref_net.multi_fc[i].weight.data
                new_fc.sigma.data=ref_net.multi_fc[i].sigma.data
                fcs.append(new_fc)
                fc1s.append(new_fc.fc1)
            net.multi_fc.fcs=nn.ModuleList(fcs)
            net.multi_fc.fc1s=nn.ModuleList(fc1s)
            net=net.to(device)

            dy_lamda=1.0*out_features/args.incre_num_classes
        elif task_id>1:
            ref_net=copy.deepcopy(net)
            ref_net=ref_net.to(device)
            ref_net.eval()
            in_features=net.multi_fc[0].in_features
            out_features1=net.multi_fc[0].fc1.out_features
            out_features2=net.multi_fc[0].fc2.out_features

            fcs=[]
            fc1s = []
            for i in range(args.fc_num):
                new_fc=resnet_pod.SplitCosineLinear(in_features,out_features1+out_features2,args.incre_num_classes).to(device)
                new_fc.fc1.weight.data[:out_features1]=ref_net.multi_fc[i].fc1.weight.data
                new_fc.fc1.weight.data[out_features1:]=ref_net.multi_fc[i].fc2.weight.data
                new_fc.sigma.data=ref_net.multi_fc[i].sigma.data
                fcs.append(new_fc)
                fc1s.append(new_fc.fc1)
            net.multi_fc.fcs = nn.ModuleList(fcs)
            net.multi_fc.fc1s = nn.ModuleList(fc1s)
            net=net.to(device)

            dy_lamda = 1.0 * (out_features1+out_features2) / args.incre_num_classes

        if task_id>0:
            cur_lamda = args.kd_loss_weight * math.sqrt(dy_lamda)
            cur_lamda_pod = args.pod_space_loss_weight * math.sqrt(dy_lamda)

        if task_id>0:
            net=net_copy.imprint_weight_pod(net,task_id,lable_real_to_logic,device,args)
            args.epochs=60

        if task_id==0:
            optim_params=net.parameters()
        else:
            ## have problems here with optimized parameters

            #ignored_params=[]
            ignored_params=list(map(id,net.multi_fc.fc1s.parameters()))
            # for i in range(args.fc_num):
            #     ignored_params.append(map(id,net.multi_fc.fcs[0].parameters()))
            learned_params=filter(lambda x:id(x) not in ignored_params, net.parameters())
            no_learned_params=filter(lambda x:id(x) in ignored_params, net.parameters())
            optim_params=[{'params':learned_params, 'lr':args.lr_initial, 'weight_decay':5e-4},
                          {'params':no_learned_params, 'lr':0, 'weight_decay':0}]


        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        optimizer = torch.optim.SGD(optim_params, lr=args.lr_initial, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[args.epochs//2, args.epochs*3//4], gamma=args.lr_decay)
        loss_CE = nn.CrossEntropyLoss().to(device)
        loss_COS = nn.CosineEmbeddingLoss().to(device)

        if args.resume_skip_base and task_id==0:
            net = torch.load('./../checkpoints/podnet/base_50_incre_10_task_0_model')
            net = net.to(device)
        else:
            net.train()
            if task_id>0:
                handle_ref_features = ref_net.multi_fc.register_forward_hook(get_ref_features)
                handle_cur_features = net.multi_fc.register_forward_hook(get_cur_features)
                handle_old_scores_bs = []
                handle_new_scores_bs = []
                for i in range(args.fc_num):
                    handle_old_scores_bs.append(net.multi_fc.fcs[i].fc1.register_forward_hook(get_old_scores_before_scale))
                    handle_new_scores_bs.append(net.multi_fc.fcs[i].fc2.register_forward_hook(get_new_scores_before_scale))
            for epoch in tqdm(range(args.epochs)):
                net.train()
                train_loss_classification=0
                train_loss_kd=0
                train_loss_mr=0
                train_loss_pod_space=0
                for (input,label) in trainloader:
                    old_scores = []
                    new_scores = []
                    input = input.to(device)
                    label = label.to(device)
                    features_new, output = net(input, features=True)
                    loss = loss_CE(output, label)
                    train_loss_classification += loss

                    if task_id > 0:
                        features_old, ref_output = ref_net(input, features=True)
                        loss_kd=loss_COS(cur_features, ref_features.detach(),torch.ones(input.shape[0],dtype=torch.float).to(device)) * cur_lamda
                        # loss_kd = loss_KL(F.log_softmax(output[:, :num_old_classes] / 2.0, dim=1),F.softmax(ref_output.detach() / 2.0,dim=1)) * 2.0 * 2.0 * 0.25 * num_old_classes

                        train_loss_kd += loss_kd
                        loss += loss_kd

                        outputs_bs = torch.cat((old_scores, new_scores), dim=1)
                        # get groud truth scores
                        gt_index = torch.zeros(outputs_bs.size()).to(device)
                        gt_index = gt_index.scatter(1, label.view(-1, 1), 1).ge(0.5)
                        gt_scores = outputs_bs.masked_select(gt_index)
                        # get top-K scores on novel classes
                        max_novel_scores = outputs_bs[:, num_old_classes:].topk(args.K, dim=1)[0]
                        # the index of hard samples, i.e., samples of old classes
                        hard_index = label.lt(num_old_classes)
                        hard_num = torch.nonzero(hard_index).size(0)
                        if hard_num > 0:
                            gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, args.K)
                            max_novel_scores = max_novel_scores[hard_index]
                            loss_mr = nn.MarginRankingLoss(margin=args.dist)(gt_scores.view(-1, 1), \
                                                                      max_novel_scores.view(-1, 1),
                                                                      torch.ones(hard_num * args.K).to(device)) * args.mr_loss_weight
                            train_loss_mr += loss_mr
                            loss += loss_mr

                        loss_pod_space=Compute_Pod_Space_Loss(features_new,features_old,device) * cur_lamda_pod
                        train_loss_pod_space+=loss_pod_space
                        loss+=loss_pod_space
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                lr_scheduler.step()
                print('epoch {} : loss_classification {} loss_kd {} loss_mr {} loss_pod_space {} '.format(epoch,train_loss_classification,train_loss_kd,train_loss_mr,train_loss_pod_space))

                old_scores = []
                new_scores = []
                net.eval()
                with torch.no_grad():
                    total_index = index_train[:args.base_num_classes + task_id * args.incre_num_classes]
                    evalset = cifar.CIFAR100(root=args.dataset_dir, train=False, transform=transform_test,
                                             index=total_index,
                                             target_transform=lable_real_to_logic)
                    evalloader = torch.utils.data.DataLoader(evalset, batch_size=100, shuffle=True, num_workers=2)
                    acc_all = Compute.compute_accuracy(net, evalloader, device)
                    print('after epoch {} : overall average accuracy is {}'.format(epoch, acc_all))

            if task_id > 0:
                handle_ref_features.remove()
                handle_cur_features.remove()
                for x in handle_old_scores_bs:
                    x.remove()
                for x in handle_new_scores_bs:
                    x.remove()



        save_path = './../checkpoints/podnet/base_{}_incre_{}_task_{}_model'.format(args.base_num_classes,args.incre_num_classes,task_id)
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
        for i in range(0,args.base_num_classes+task_id*args.incre_num_classes,1):
            data=cifar.CIFAR100(root=args.dataset_dir, train=True, transform=transform_train, index=[index_train[i]],target_transform=lable_real_to_logic)
            loader=torch.utils.data.DataLoader(data,batch_size=100, shuffle=False, num_workers=2)
            feature=Compute_Feature_podnet(loader,net,device,data.__len__())
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