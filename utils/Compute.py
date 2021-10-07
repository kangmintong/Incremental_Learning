import torch
import numpy as np
import torch.nn.functional as F
import torch_dct
from scipy.spatial.distance import cdist

def compute_accuracy(net,test_loader,device):
    corrct=0
    all=0
    for (input,label) in test_loader:
        input=input.to(device)
        label=label.to(device)
        output=net(input)
        all += len(label)
        corrct += torch.eq(torch.max(output,dim=1)[1],label).sum().item()
    return 1.0*corrct/all

def compute_accuracy_cross(net,ref_net, w, test_loader,device):
    corrct=0
    all=0
    for (input,label) in test_loader:
        input=input.to(device)
        label=label.to(device)
        output=net.cross_forward(input, ref_net, w)
        all += len(label)
        corrct += torch.eq(torch.max(output,dim=1)[1],label).sum().item()
    return 1.0*corrct/all

def Compute_One_Hot(y,k,device):
    ret=torch.zeros((y.__len__(),k))
    for i in range(len(y)):
        ret[i][y[i]]=1
    ret=ret.to(device)
    return ret

def Compute_Feature(loader,net,device,num_data):
    net.eval()
    net=net.to(device)
    num_feature=net.fc.in_features
    features = torch.zeros([num_data,num_feature]).to(device)
    cur=0
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(device)
            _,_,fea=net(x,feature_output=True)
            features[cur:cur+len(x),:]=fea
            cur+=len(x)
    return features.detach().cpu().numpy()

def Compute_Feature_lucir(loader,net,device,num_data):
    net.eval()
    net=net.to(device)
    num_feature=net.fc.in_features
    features = torch.zeros([num_data,num_feature]).to(device)
    cur=0
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(device)
            fea=net(x,True)
            features[cur:cur+len(x),:]=fea
            cur+=len(x)
    return features.detach().cpu().numpy()

def Compute_Feature_podnet(loader,net,device,num_data):
    net.eval()
    net=net.to(device)
    num_feature=net.multi_fc.fcs[0].in_features
    features = torch.zeros([num_data,num_feature]).to(device)
    cur=0
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(device)
            fea=net(x,features=False,last_feature=True)
            features[cur:cur+len(x),:]=fea
            cur+=len(x)
    return features.detach().cpu().numpy()

def Compute_Feature_ours(loader,net,device,num_data):
    net.eval()
    net=net.to(device)
    num_feature=net.fc.in_features
    features = torch.zeros([num_data,num_feature]).to(device)
    cur=0
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(device)
            fea=net(x,features=False,last_feature=True)
            features[cur:cur+len(x),:]=fea
            cur+=len(x)
    return features.detach().cpu().numpy()

def Compute_Feature_cross(loader,net,ref_net,w,device,num_data):
    net.eval()
    net=net.to(device)
    num_feature=net.fc.in_features
    features = torch.zeros([num_data,num_feature]).to(device)
    cur=0
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(device)
            fea=net.cross_forward(x,ref_net,w,features=False,last_feature=True)
            features[cur:cur+len(x),:]=fea
            cur+=len(x)
    return features.detach().cpu().numpy()

def Compute_Pod_Space_Loss(features_new,features_old,device):
    loss = torch.tensor(0.).to(device)
    for a,b in zip(features_new, features_old):
        a=torch.pow(a,2)
        b=torch.pow(b,2)
        a_h=a.sum(dim=3).view(a.shape[0],-1)
        b_h=b.sum(dim=3).view(b.shape[0],-1)
        a_w=a.sum(dim=2).view(a.shape[0],-1)
        b_w=b.sum(dim=2).view(b.shape[0],-1)
        a=torch.cat([a_h,a_w],dim=-1)
        b=torch.cat([b_h,b_w],dim=-1)
        a=F.normalize(a,p=2,dim=-1)
        b=F.normalize(b,p=2,dim=-1)
        cur_loss=torch.mean((a-b).norm(p=2, dim=-1))
        loss+=cur_loss
    return loss/len(features_new)

def Compute_Ours_Freq_Loss(features_new,features_old,weights,device):
    loss = torch.tensor(0.).to(device)
    cnt=0
    for a,b in zip(features_new, features_old):
        a=torch_dct.dct_3d(a)[:,:,:,:].contiguous().view(a.shape[0],-1)
        b=torch_dct.dct_3d(b)[:,:,:,:].contiguous().view(b.shape[0],-1)
        a=F.normalize(a,p=2,dim=-1)
        b=F.normalize(b,p=2,dim=-1)
        cur_loss=torch.mean((a-b).norm(p=2,dim=-1))
        loss+=cur_loss*weights[cnt]
        cnt+=1
    return loss/len(features_new)

def compute_accuracy_ncm_ours(net, test_loader, class_means, device):
    correct = 0
    all = 0
    class_means = np.array(class_means)
    for (input, label) in test_loader:
        input = input.to(device)
        label = label.to(device)
        output = net(input,features=False, last_feature=True)
        all += len(label)

        euc_ncm = cdist(np.array(class_means), np.array(output), 'sqeuclidean')
        score_ncm = torch.from_numpy((-euc_ncm).T).to(device)
        _, predicted_ncm = score_ncm.max(1)
        correct += predicted_ncm.eq(label).sum().item()
    return 1.0 * correct / all

def get_se_layers(net):
    return 1

def compute_se_attention(net, refnet):
    net_se_layers = get_se_layers(net)
    refnet_se_layers = get_se_layers(refnet)
    loss=torch.tensor([0.])
    for a,b in zip(net_se_layers, refnet_se_layers):
        loss+=torch.mean((a-b).norm(dim=-1,p=1))
    return loss

