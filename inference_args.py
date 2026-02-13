from torch import nn
import torch
import numpy as np
from dataset.body_ct import DualStackSampler
from model.insertnet_v1 import InsertionNet
import torchvision
import os
import argparse
import json
import nibabel as nib
import albumentations as A
from nifti2npy import Nifti2Npy
from scipy.ndimage import gaussian_filter
from chart.chart import Chart, ChartGroup
import matplotlib.pyplot as plt

SLICE_NUM = 256

def prepare_network_input(im1, im2):

    def sample_slices(num_slices, num_sampled, voxel_size):

        all_indices = np.array(list(range(num_slices)))
        if num_slices < num_sampled:
            return all_indices, voxel_size
        
        # gap sampling
        k = int(np.ceil(num_slices / num_sampled))
        indices = all_indices[::k]
        voxel_size[2] = voxel_size[2] * k
        return indices, voxel_size

    data1 = np.array(im1.dataobj)
    data2 = np.array(im2.dataobj)

    vox1 = np.abs([im1.affine[0, 0], im1.affine[1, 1], im1.affine[2, 2]])
    vox2 = np.abs([im2.affine[0, 0], im2.affine[1, 1], im2.affine[2, 2]])

    # sample slices
    indices1, vox1 = sample_slices(data1.shape[2], SLICE_NUM-2, vox1)
    indices2, vox2 = sample_slices(data2.shape[2], SLICE_NUM, vox2)

    stack1 = np.zeros((256, 256, SLICE_NUM))
    stack2 = np.zeros((256, 256, SLICE_NUM))
    mask1 = np.zeros(SLICE_NUM)
    mask2 = np.zeros(SLICE_NUM)

    # starting dumb slice
    stack1[:, :, 0] = 1
    # ending dumb slice
    stack1[:, :, len(indices1) + 1] = -1

    stack1[:, :, 1: len(indices1) + 1] = data1[:, :, indices1]
    stack2[:, :, : len(indices2)] = data2[:, :, indices2]
    mask1[: len(indices1) + 2] = 1
    mask2[: len(indices2)] = 1

    res = {
        'stack1': stack1.astype(np.float32),
        'stack2': stack2.astype(np.float32),
        'mask1': mask1.astype(np.bool_), 
        'mask2': mask2.astype(np.bool_), 
        'vox1': vox1,
        'vox2': vox2,
    }

    return res

def preprocessing(nii_path):

    img = nib.load(nii_path)
    affine = img.affine
    
    data = img.get_fdata()
    
    p = Nifti2Npy()
    data = p.rescale_xy(data)
    resize = A.Resize(256, 256)

    new_affine = affine.copy()
    new_affine[0, 0] = affine[0, 0] * (img.shape[0] / 256)
    new_affine[1, 1] = affine[1, 1] * (img.shape[1] / 256)
    
    
    blurred = gaussian_filter(data, (0.8, 0.8, 0), truncate=3)
    resized = resize(image=blurred)['image']
    resized = resized.astype(np.float32)

    return nib.Nifti1Image(resized, new_affine)

parser = argparse.ArgumentParser(description="Process input arguments.")
parser.add_argument("--test_nii", type=str, help="Test file path")
parser.add_argument("--template_nii", type=str, help="Template file path")
parser.add_argument("--device", type=str, help="Device to run", default='cuda')

args = parser.parse_args()

device = args.device

im_tokenizer = torchvision.models.resnet18()
im_tokenizer.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
im_tokenizer.fc = torch.nn.Identity()

model = InsertionNet(
        im_tokenizer=im_tokenizer,
        d_model=512,
        max_seq_len=SLICE_NUM,
        n_head=8,
        n_enc_layers=2,
        n_dec_layers=2
    )

model.load_state_dict(torch.load('exp/Exp_body_ct_semi_supervised_1.0_kl_sinPE_1/checkpoints/best.pth', map_location='cpu')['model'])

test_path = args.test_nii
template_path = args.template_nii

template_image = preprocessing(template_path)
test_image = preprocessing(test_path)

input_data_np = prepare_network_input(test_image, template_image)
input_data = {}
for k, v in input_data_np.items():
    input_data[k] = torch.from_numpy(v).unsqueeze(0).to(device)

model.to(device)
model.eval()

i = 0
stack1 = input_data['stack1'][i: i + 1].permute(3, 0, 1, 2)
stack2 = input_data['stack2'][i: i + 1].permute(3, 0, 1, 2)
mask1 = input_data['mask1'][i: i + 1]
mask2 = input_data['mask2'][i: i + 1]
vox1 = input_data['vox1'][i]
vox2 = input_data['vox2'][i]

with torch.no_grad():
    atten_2to1_before_softmax = model(stack1, stack2, mask1, mask2, vox1, vox2)

valid_attn = atten_2to1_before_softmax[0, 0, : sum(mask2[0]), : sum(mask1[0])]

template_key_slices = {'2': 204, '3': 190, '4': 159, '5': 141, '6': 129, '7': 94, '8': 25}

cg = ChartGroup(1, 4)
c1 = cg.get_chart(1, 1)
c2 = cg.get_chart(1, 2)
c3 = cg.get_chart(1, 3)
c4 = cg.get_chart(1, 4)
c3.config(18, 18)
c4.config(18, 18)

c1.axis_off().slice(input_data['stack1'][0].detach().cpu().numpy(), 
                    orientation='y', voxsz=input_data['vox1'][0].detach().cpu().numpy())
c2.axis_off().slice(input_data['stack2'][0].detach().cpu().numpy(), 
                    orientation='y', voxsz=input_data['vox2'][0].detach().cpu().numpy())

for i, index2 in enumerate([20, 60, 100, 150, 200]):
    # index2 = template_key_slices[str(ks)]
    index_2in1 = valid_attn.argmax(axis=1).detach().cpu().numpy()[index2]

    c1.line([0, 256], [index_2in1, index_2in1], color=Chart.color_scheme2[i], linewidth=3)
    c2.line([0, 256], [index2, index2], color=Chart.color_scheme2[i], linewidth=3)
    
    pred_dist = torch.softmax(atten_2to1_before_softmax[0, 0], axis=1)
    x = np.array(list(range(SLICE_NUM)))
    c3.line(x, pred_dist[index2].detach().cpu().numpy(), color=Chart.color_scheme2[i], linewidth=3)

im = c4.ax.imshow(pred_dist.detach().cpu().numpy(), cmap='Blues', origin='lower')
cbar = plt.colorbar(im, ax=c4.ax, label='Value', orientation='horizontal')
cbar.ax.tick_params(labelsize=14)

cg.show()

print(1)