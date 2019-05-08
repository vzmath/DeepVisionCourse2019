from PIL import Image
from matplotlib import pyplot as plt
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from torch import topk
from tqdm import tqdm
import torch
import numpy as np
import skimage.transform
import os
from dataset import MIPDataset

# saved model and dest dir for saliency maps
cam_dir = '/home/vince/programming/python/research/rail/CTA-project/3D/saliency_maps'

# MIPs generation parameters
neg_test_case = '../data/processed/test/control'
pos_test_case = '../data/processed/test/occlusion'
excel_path = '/home/vince/programming/python/research/rail/CTA-project/data/processed/head_indices.xlsx'
control_sheet = 'control'
occl_sheet = 'occlusion'
cols = ['patient id', 'z_start', 'z_end']
THICKNESS = 20

if not os.path.exists(cam_dir):
    os.mkdir(cam_dir)
num_classes = 2
model_name = 'model_oversampling.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SaveFeatures():
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

def main():
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),])

    display_transform = transforms.Compose([
        transforms.Resize(224)])

    # generate test MIPs on the fly
    neg_mips_testset = MIPDataset(neg_test_case).get_mips_dataset((excel_path, control_sheet, cols), THICKNESS, False, 'z')
    pos_mips_testset = MIPDataset(pos_test_case).get_mips_dataset((excel_path, occl_sheet, cols), THICKNESS, True, 'z')
    mips_testset = {**neg_mips_testset, **pos_mips_testset}     # only applicable in Python 3.5 or later

    for mip_info, (mip, mip_label) in tqdm( mips_testset.items() ):
        trained_model = models.resnet152()
        trained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_features = trained_model.fc.in_features
        trained_model.fc = nn.Linear(num_features, num_classes)
        trained_model.load_state_dict(torch.load(model_name))
        trained_model.eval()
        trained_model.to(device)
        final_layer = trained_model._modules.get('layer4')
        activated_features = SaveFeatures(final_layer)

        test_mip = Image.fromarray(mip)
        test_mip = test_mip.convert('RGB')      # convert grayscale to RGB for line 76
        tensor = preprocess(test_mip)
        prediction_var = Variable((tensor.unsqueeze(0)).to(device), requires_grad=True)
        prediction = trained_model(prediction_var)
        pred_probas = F.softmax(prediction).data.squeeze()
        activated_features.remove()
        #print(topk(pred_probas, 1))

        weight_softmax_params = list(trained_model._modules.get('fc').parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
        #print(weight_softmax_params)
        class_idx = topk(pred_probas,1)[1].int()
        overlay = getCAM(activated_features.features, weight_softmax, class_idx )
        plt.imshow(display_transform(test_mip))
        plt.imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.5, cmap='jet')
        cam_filename = '{}_{}_{}_axis_{}_label_{}.jpeg'.format(mip_info[3], mip_info[0], mip_info[1], mip_info[2], mip_label)
        plt.savefig(os.path.join(cam_dir, cam_filename))

if __name__ == "__main__":
    main()
