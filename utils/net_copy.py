import torch
from data import cifar
import torchvision.transforms as transforms
from utils import Compute
import torch.nn.functional as F
from sklearn.cluster import KMeans

def imprint_weight(net, task_id, label_real_to_logic, device, args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
    ])
    net.eval()
    ave_old_fc=torch.mean(net.fc.fc1.weight.data.norm(dim=1,keepdim=True), dim=0).to('cpu')
    novel_fc = torch.zeros((args.incre_num_classes, net.fc.in_features))
    for i in range(args.base_num_classes+(task_id-1)*args.incre_num_classes, args.base_num_classes+task_id*args.incre_num_classes):
        trainset = cifar.CIFAR100(root=args.dataset_dir, train=True, transform=transform_train, index=[i],
                                  target_transform=label_real_to_logic)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        features = Compute.Compute_Feature_lucir(trainloader, net, device, trainset.__len__())
        features = F.normalize(torch.from_numpy(features), p=2, dim=1)
        feature = torch.mean(features, dim=0)
        novel_fc[i - args.base_num_classes-(task_id-1)*args.incre_num_classes] = F.normalize(feature,p=2,dim=0)*ave_old_fc
    net.fc.fc2.weight.data = novel_fc.to(device)
    net=net.to(device)
    return net

def imprint_weight_ours(net, task_id, label_real_to_logic, device, args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
    ])
    net.eval()
    ave_old_fc=torch.mean(net.fc.fc1.weight.data.norm(dim=1,keepdim=True), dim=0).to('cpu')
    novel_fc = torch.zeros((args.incre_num_classes, net.fc.in_features))
    for i in range(args.base_num_classes+(task_id-1)*args.incre_num_classes, args.base_num_classes+task_id*args.incre_num_classes):
        trainset = cifar.CIFAR100(root=args.dataset_dir, train=True, transform=transform_train, index=[i],
                                  target_transform=label_real_to_logic)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        features = Compute.Compute_Feature_ours(trainloader, net, device, trainset.__len__())
        features = F.normalize(torch.from_numpy(features), p=2, dim=1)
        feature = torch.mean(features, dim=0)
        novel_fc[i - args.base_num_classes-(task_id-1)*args.incre_num_classes] = F.normalize(feature,p=2,dim=0)*ave_old_fc
    net.fc.fc2.weight.data = novel_fc.to(device)
    net=net.to(device)
    return net

def imprint_weight_pod(net, task_id, label_real_to_logic, device, args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
    ])
    net.eval()
    # ave_old_fc=torch.mean(net.fc.fc1.weight.data.norm(dim=1,keepdim=True), dim=0).to('cpu')
    # novel_fc = torch.zeros((args.incre_num_classes, net.fcs[0].in_features))
    for i in range(args.base_num_classes+(task_id-1)*args.incre_num_classes, args.base_num_classes+task_id*args.incre_num_classes):
        trainset = cifar.CIFAR100(root=args.dataset_dir, train=True, transform=transform_train, index=[i],
                                  target_transform=label_real_to_logic)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        features = Compute.Compute_Feature_podnet(trainloader, net, device, trainset.__len__())
        features = F.normalize(torch.from_numpy(features), p=2, dim=1)

        kmeans = KMeans(n_clusters=args.fc_num, random_state=1993).fit(features)
        centers = torch.tensor(kmeans.cluster_centers_)
        centers = F.normalize(centers, p=2, dim=1)

        for j in range(args.fc_num):
            net.multi_fc.fcs[j].fc2.weight.data[i - args.base_num_classes-(task_id-1)*args.incre_num_classes]=centers[j].to(device)

        #feature = torch.mean(features, dim=0)
        #novel_fc[i - args.base_num_classes-(task_id-1)*args.incre_num_classes] = F.normalize(feature,p=2,dim=0)*ave_old_fc

    net=net.to(device)
    return net

