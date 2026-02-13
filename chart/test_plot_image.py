import pandas as pd
import nibabel as nib
import json
from chart import Chart, ChartGroup
import numpy as np

img = nib.load('E:/dingjie/mipresearch/body_part_regression/thomas/dataset/unique_subject/images_DS/1.2.840.113654.2.70.1.105618831923228527027225985746361996970.nii.gz')
img_data = img.get_fdata().astype(np.float32)

scores_gt = json.load(open('score_assignment_all_slices.json'))
ks_labels = json.load(open('all_key_slice_labels_pure.json'))
dataset = json.load(open('E:/dingjie/mipresearch/body_part_regression/bpr_new/dataset.json'))

