import sys
sys.path.append('..')
import data.cifar as cifar
import torchvision.transforms as transforms
import torch
import argparse
from scipy.spatial.distance import cdist
from utils.Compute import Compute_Feature_ours
from utils.Compute import Compute_Ours_Freq_Loss
import utils.Compute as Compute
import numpy as np


def eval_ours_ncm(args):
    if args.dataset=='cifar100':
        total_num_classes=100
    device=torch.device(args.gpu_info if torch.cuda.is_available() else "cpu")
    num_tasks=(total_num_classes-args.base_num_classes)//args.incre_num_classes

    index_train=torch.load(args.store_path+'index_train')
    lable_real_to_logic={}
    for i in range(index_train.__len__()):
        lable_real_to_logic[index_train[i]]=i
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023))
    ])

    ave_acc=[]
    with torch.no_grad():
        for task_id in range(num_tasks+1):
            class_means = torch.load(args.store_path+'class_means_task_{}'.format(task_id))
            net = torch.load(args.store_path+'base_{}_incre_{}_task_{}_model'.format(args.base_num_classes, args.incre_num_classes,task_id))
            net = net.to(device)
            net.eval()
            acc = []
            for previous_id in range(task_id + 1):
                if previous_id == 0:
                    index_pre = index_train[:args.base_num_classes]
                else:
                    index_pre = index_train[args.base_num_classes + (
                            previous_id - 1) * args.incre_num_classes:args.base_num_classes + previous_id * args.incre_num_classes]
                evalset_pre = cifar.CIFAR100(root=args.dataset_dir, train=False, transform=transform_test, index=index_pre,
                                             target_transform=lable_real_to_logic)
                evalloader_pre = torch.utils.data.DataLoader(evalset_pre, batch_size=100, shuffle=True, num_workers=2)
                acc_pre = Compute.compute_accuracy_ncm_ours(net, evalloader_pre, class_means, device)
                print('task {} : ncm test accuracy : {} '.format(previous_id, acc_pre))
                acc.append(acc_pre)

            total_index = index_train[:args.base_num_classes + task_id * args.incre_num_classes]
            evalset = cifar.CIFAR100(root=args.dataset_dir, train=False, transform=transform_test, index=total_index,
                                     target_transform=lable_real_to_logic)
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=100, shuffle=True, num_workers=2)
            acc_all = Compute.compute_accuracy_ncm_ours(net, evalloader, class_means, device)
            print('after training {} : NCM overall average accuracy is {}'.format(task_id, acc_all))
            print('')
            ave_acc.append(acc_all)
    print('average accuracy is {} '.format(np.mean(ave_acc)))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-dataset', type=str, default='cifar100')
    parser.add_argument('-dataset_dir', type=str, default='./../datasets/cifar')
    parser.add_argument('-seed', type=int, default=1993)
    parser.add_argument('-base_num_classes', type=int, default=50)
    parser.add_argument('-incre_num_classes', type=int, default=10)
    parser.add_argument('-gpu_info', type=str, default='cuda:0')
    parser.add_argument('-store_path', type=str, default='./../checkpoints/ours')
    args = parser.parse_args()
    eval_ours_ncm(args)