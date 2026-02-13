from torch_trainer import Trainer
from torch import nn
import torch
import random
import numpy as np
from dataset.body_ct import DualStackSamplerSemiSupervised
from model.insertnet_v1 import InsertionNet
import torchvision
import os
from torch.utils.data import DataLoader
import argparse
from torch.optim import Adam
import json
from dataset.volume_augmentation_torch import augment
import sys

BATCH_SIZE = 1
SLICE_NUM = 256

class MyTrainer(Trainer):

    def train_loss(self, model, input_data):

        total_loss = 0

        for i in range(BATCH_SIZE):

            stack1 = input_data['stack1'][i: i + 1]
            stack2 = input_data['stack2'][i: i + 1]
            mask1 = input_data['mask1'][i: i + 1]
            mask2 = input_data['mask2'][i: i + 1]
            vox1 = input_data['vox1'][i]
            vox2 = input_data['vox2'][i]

            valid_len1 = mask1.sum()
            valid_len2 = mask2.sum()

            stack1_aug = augment(stack1[:, :, :, 1: valid_len1 - 1][None], spacing=vox1)[0]
            stack2_aug = augment(stack2[:, :, :, : valid_len2][None], spacing=vox2)[0]

            stack1[:, :, :, 1: valid_len1 - 1] = stack1_aug
            stack2[:, :, :, : valid_len2] = stack2_aug

            # import nibabel as nib
            # affine1 = np.eye(4)
            # affine1[0, 0] = vox1[0]
            # affine1[1, 1] = vox1[1]
            # affine1[2, 2] = vox1[2]
            # nib.save(nib.Nifti1Image(stack1.cpu().numpy()[0], affine1), 'temp1.nii.gz')

            # affine2 = np.eye(4)
            # affine2[0, 0] = vox2[0]
            # affine2[1, 1] = vox2[1]
            # affine2[2, 2] = vox2[2]
            # nib.save(nib.Nifti1Image(stack2.cpu().numpy()[0], affine2), 'temp2.nii.gz')

            atten_2to1_before_softmax = model(stack1.permute(3, 0, 1, 2), 
                                              stack2.permute(3, 0, 1, 2), 
                                              mask1, mask2, 
                                              vox1, vox2)

            valid_attn = atten_2to1_before_softmax[i, 0, : sum(mask2[0]), : sum(mask1[0])]
            dist_gt = input_data['index_2in1_gaussian'][i, : sum(mask2[0]), : sum(mask1[0])]

            valid_attn_softmax = torch.softmax(valid_attn, dim=1)

            loss_kl = torch.sum(dist_gt * -torch.log_softmax(valid_attn, dim=1), dim=1)

            loss = loss_kl
        
            total_loss += loss.mean()

        loss_info = {
            'total_loss': total_loss / BATCH_SIZE
        }
            
        return {}, loss_info


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

dataset = json.load(open('dataset.json'))
gt_scores = json.load(open('score_assignment_all_slices.json'))
data_root = '/workspace/data/partial_volumes_preprocessed_256'

semi_r = float(sys.argv[1])
all_supervised_cases = list(gt_scores)
rand_supervised_ind = np.random.permutation(len(all_supervised_cases))[: int(semi_r * len(all_supervised_cases))]
supervised_cases = [all_supervised_cases[i] for i in rand_supervised_ind]
gt_scores = dict([(k, v) for k, v in gt_scores.items() if k in supervised_cases])

print('Use supervisded N=', len(gt_scores))


train_dataset = DualStackSamplerSemiSupervised(
    data_root=data_root, 
    cases=dataset['labeled_train'], 
    gt_scores=gt_scores, 
    slice_size=256,
    slice_num=SLICE_NUM,
    )

val_dataset = DualStackSamplerSemiSupervised(
    data_root=data_root, 
    cases=dataset['val'], 
    gt_scores=gt_scores, 
    slice_size=256,
    slice_num=SLICE_NUM,
    )

train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=0)

val_loader = DataLoader(dataset=val_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=False, num_workers=0)

trainer = MyTrainer(save_dir='exp', hparam={'num_training': len(train_dataset),'num_val': 0}, name=f'body_ct_semi_supervised_{semi_r}_kl_sinPE')

optimizer = Adam(params=model.parameters(), lr=1e-4)

gpu = int(sys.argv[2])
device = torch.device('cuda:%s' % gpu)

trainer.fit_and_val(
    model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, 
    device=device, save_per_epoch=10, save_best=True,
    total_epochs=100, 
    log_per_iteration=10
)
