import argparse
# import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import copy
import torch.nn.functional as F


def grad_cam_loss(transform_net_old,transform_net_new,feature_old,feature_new,device):

    feature_new=feature_new.requires_grad_(True)
    out_n=transform_net_new(feature_new)

    batch = out_n.size()[0]
    index = out_n.argmax(dim=-1).view(-1, 1)
    onehot = torch.zeros_like(out_n)
    onehot.scatter_(-1, index, 1.)
    out_n = torch.sum(onehot * out_n)
    out_n.backward()
    grads_n = transform_net_new.gradients[-1]

    feature_old = feature_old.requires_grad_(True)
    out_o = transform_net_old(feature_old)
    out_o=torch.sum(onehot*out_o)
    out_o.backward()
    grads_o=transform_net_old.gradients[-1]

    # print('o')
    # print(grads_o[0][0][0])
    # print('n')
    # print(grads_n[0][0][0])

    weight_o = grads_o.mean(dim=1).view(batch, -1, 1, 1)
    weight_n = grads_n.mean(dim=1).view(batch, -1, 1, 1)

    cam_o = F.relu((grads_o * weight_o).sum(dim=1))
    cam_n = F.relu((grads_n * weight_n).sum(dim=1))
    # normalization
    cam_o = F.normalize(cam_o.view(batch, -1), p=2, dim=-1)
    cam_n = F.normalize(cam_n.view(batch, -1), p=2, dim=-1)
    loss_AD = (cam_o - cam_n).norm(p=1, dim=1).mean()
    return loss_AD

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img)
    #return np.uint8(img * 255)

def get_gradcam(img,net,usecuda=True):
    grad_cam = GradCam(model=net, feature_module=net.layer3, target_layer_names=["4"], use_cuda=usecuda)
    input=img.clone()
    input.unsqueeze_(0)
    input = input.requires_grad_(True)
    target_index = None
    mask = grad_cam(input, target_index)
    # gb_model = GuidedBackpropReLUModel(model=net, use_cuda=usecuda)
    # gb = gb_model(input, index=target_index)
    # gb = gb.transpose((1, 2, 0))
    # cam_mask = cv2.merge([mask, mask, mask])
    # cam_gb = deprocess_image(cam_mask * gb)
    cam_gb=mask
    return cam_gb



# My version

# import sys
# sys.path.append('..')
# from network import resnet
# import torch
# from torchvision import transforms
# from data import cifar
# import numpy as np
# from utils import image_read_save
# import cv2
# from torch.autograd import Function
#
#
# class FeatureExtractor():
#     def __init__(self,feature_module,target_layer_names):
#         self.feature_module=feature_module
#         self.target_layer_names=target_layer_names
#         self.gradients=[]
#     def save_gradients(self,grad):
#         self.gradients.append(grad)
#     def __call__(self, x):
#         outputs=[]
#         self.gradients=[]
#         for name,module in self.feature_module._modules.items():
#             x=module(x)
#             if name in self.target_layer_names:
#                 x.register_hook(self.save_gradients)
#                 outputs+=[x]
#         return outputs,x
#
# class ModelOutputs():
#     def __init__(self,model,feature_module,target_layer_names):
#         self.model=model
#         self.feature_module=feature_module
#         self.extractor=FeatureExtractor(feature_module,target_layer_names)
#     def get_gradients(self):
#         return self.extractor.gradients
#     def __call__(self,x):
#         target_activations=[]
#         for name,module in self.model._modules.items():
#             if module==self.feature_module:
#                 target_activations,x=self.extractor(x)
#             elif 'avgpool' in name.lower():
#                 x=module(x)
#                 x=x.view(x.size(0),-1)
#             else:
#                 x=module(x)
#         return target_activations,x
#
# class GradCam():
#     def __init__(self,model,feature_module,target_layer_names,device):
#         model.eval()
#         self.model=model
#         self.device=device
#         self.model=self.model.to(device)
#         self.feature_module=feature_module
#         self.target_layer_names=target_layer_names
#         self.extractor=ModelOutputs(self.model,self.feature_module,self.target_layer_names)
#     def forward(self,x):
#         return self.model(x)
#     def __call__(self, x, index=None):
#         x=x.to(self.device)
#         features,output=self.extractor(x)
#         if index==None:
#             index=np.argmax(output.cpu().data.numpy())
#
#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0][index] = 1
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         one_hot = one_hot.to(self.device)
#         one_hot = torch.sum(one_hot * output)
#         self.feature_module.zero_grad()
#         self.model.zero_grad()
#         one_hot.backward(retain_graph=True)
#         grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
#
#         target = features[-1]
#         target = target.cpu().data.numpy()[0, :]
#         weights = np.mean(grads_val, axis=(2, 3))[0, :]
#         cam = np.zeros(target.shape[1:], dtype=np.float32)
#         for i, w in enumerate(weights):
#             cam += w * target[i, :, :]
#         cam = np.maximum(cam, 0)
#
#         cam = cv2.resize(cam, x.shape[2:])
#         cam = cam - np.min(cam)
#         cam = cam / np.max(cam)
#         return cam
#
# class GuidedBackpropReLU(Function):
#
#     @staticmethod
#     def forward(self, input):
#         positive_mask = (input > 0).type_as(input)
#         output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
#         self.save_for_backward(input, output)
#         return output
#
#     @staticmethod
#     def backward(self, grad_output):
#         input, output = self.saved_tensors
#         grad_input = None
#
#         positive_mask_1 = (input > 0).type_as(grad_output)
#         positive_mask_2 = (grad_output > 0).type_as(grad_output)
#         grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
#                                    torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
#                                                  positive_mask_1), positive_mask_2)
#
#         return grad_input
#
# class GuidedBackpropReLUModel:
#     def __init__(self, model, device):
#         self.model = model
#         self.model.eval()
#         self.device=device
#         self.model=self.model.to(device)
#
#         def recursive_relu_apply(module_top):
#             for idx, module in module_top._modules.items():
#                 recursive_relu_apply(module)
#                 if module.__class__.__name__ == 'ReLU':
#                     module_top._modules[idx] = GuidedBackpropReLU.apply
#
#         # replace ReLU with GuidedBackpropReLU
#         recursive_relu_apply(self.model)
#
#     def forward(self, input):
#         return self.model(input)
#
#     def __call__(self, input, index=None):
#         input=input.to(self.device)
#         input=input.requires_grad_(True)
#         output=self.forward(input)
#         if index == None:
#             index = np.argmax(output.cpu().data.numpy())
#
#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0][index] = 1
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         one_hot=one_hot.to(self.device)
#         one_hot = torch.sum(one_hot * output)
#         input.retain_grad()
#         one_hot.backward(retain_graph=True)
#         output = input.grad.cpu().data.numpy()
#         output = output[0, :, :, :]
#         return output
#
# def deprocess_image(img):
#     """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
#     img = img - np.mean(img)
#     img = img / (np.std(img) + 1e-5)
#     img = img * 0.1
#     img = img + 0.5
#     img = np.clip(img, 0, 1)
#     return np.uint8(img)
#
# def get_gradcam(img,net,device):
#     grad_cam = GradCam(model=net, feature_module=net.layer3, target_layer_names=["4"], device=device)
#     input = img.unsqueeze(0)
#     input = input.requires_grad_(True)
#     mask = grad_cam(input, None)
#     gb_model = GuidedBackpropReLUModel(model=net, device=device)
#     gb = gb_model(input, index=None)
#     gb = gb.transpose((1, 2, 0))
#     cam_mask = cv2.merge([mask, mask, mask])
#     gb = deprocess_image(gb)
#     cam_gb = deprocess_image(cam_mask * gb)
#     cam_gb = torch.tensor(cam_gb).to(device)
#     return cam_gb
#
#
# # below are test codes
# # net=resnet.resnet32(num_classes=50)
# # net=torch.load('/home/mintong/iccv2021/IL/checkpoints/lwf/base_10_incre_10_task_0_model')
# # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # transform_train = transforms.Compose([
# #     transforms.RandomCrop(32, padding=4),
# #     transforms.RandomHorizontalFlip(),
# #     transforms.ToTensor(),
# #     transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023))
# # ])
# # np.random.seed(1993)
# # index_train=np.array(range(100))
# # np.random.shuffle(index_train)
# # lable_real_to_logic={}
# # for i in range(index_train.__len__()):
# #     lable_real_to_logic[index_train[i]]=i
# # index_cur=index_train[:50]
# # trainset = cifar.CIFAR100(root='./../datasets/cifar', train=True, transform=transform_train, index=index_cur, target_transform=lable_real_to_logic)
# #
# # grad_cam = GradCam(model=net, feature_module=net.layer3,target_layer_names=["4"],device=device)
# # input=[]
# # input.append(trainset[0][0])
# # input.append(trainset[1][0])
# #
# #
# # x=get_gradcam(trainset[0][0],net,device)
# # x=get_gradcam(trainset[0][0],net,device)
#
# # mask=grad_cam(input,None)
# # gb_model = GuidedBackpropReLUModel(model=net, device=device)
# # gb = gb_model(input, index=None)
# # gb = gb.transpose((1, 2, 0))
# # cam_mask = cv2.merge([mask, mask, mask])
# # gb = deprocess_image(gb)
# # cam_gb = deprocess_image(cam_mask*gb)
#
# # image_read_save.save_cam(torch.tensor(mask),'./../visualization/mask.jpg')
# # cv2.imwrite('./../visualization/gb.jpg',gb)
# # cv2.imwrite('./../visualization/cam_gb.jpg',cam_gb)
