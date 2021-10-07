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

def Compute_Attention_Loss(net, ref_net, output, ref_output, feature_new, feature_old, device):
    batch = len(output)
    num_old_classes = ref_output.shape[1]
    index = output[:, :num_old_classes].argmax(dim=-1).view(-1, 1)
    index = index.requires_grad_(False)

    onehot_old = torch.zeros_like(ref_output)
    onehot_old.scatter_(-1, index, 1.)
    out_o = torch.sum(onehot_old * ref_output)
    out_o.backward(retain_graph=True)
    grads_o_list = [ref_net.layer1.gradients[-1], ref_net.layer2.gradients[-1], ref_net.layer3.gradients[-1], ref_net.gradients[-1]]

    onehot_new = torch.zeros_like(output)
    onehot_new.scatter_(-1, index, 1.)
    out_n = torch.sum(onehot_new * output)
    out_n.backward(retain_graph=True)
    grads_n_list = [net.layer1.gradients[-1], net.layer2.gradients[-1], net.layer3.gradients[-1], net.gradients[-1]]
    loss = torch.tensor(0.).to(device)

    for grads_o, grads_n, f_o, f_n in zip(grads_o_list, grads_n_list, feature_old, feature_new):
        weight_o = torch.mean(grads_o, dim=1, keepdim=True)
        weight_n = torch.mean(grads_n, dim=1, keepdim=True)

        cam_o = torch.sum(f_o * weight_o, dim=1).squeeze()
        cam_n = torch.sum(f_n * weight_n, dim=1).squeeze()
        cam_o = F.normalize(cam_o.view(batch, -1), p=2, dim=-1)
        cam_n = F.normalize(cam_n.view(batch, -1), p=2, dim=-1)

        cam_o = cam_o.masked_select(cam_o.gt(0.3))
        cam_n = cam_n.masked_select(cam_n.gt(0.3))
        loss_gradcam = F.normalize(torch.abs(cam_n - cam_o), p=2, dim=1).mean()
        loss+=loss_gradcam
    return loss/len(feature_new)