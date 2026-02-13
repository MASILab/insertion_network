import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
# from monai.transforms import RandSpatialCrop, Resize, RandRotate
# from scipy.interpolate import interp1d
from scipy.stats import norm

class DualStackSampler(Dataset):

    def __init__(self, data_root, cases, fixed_case2=None, gt_scores=None, paired_cases=None, sampling_method=None,
                 stack_range=(0.3, 1), slice_num=256, slice_size=256):
        self.data_root = data_root
        self.cases = cases
        self.fixed_case2 = fixed_case2
        self.stack_range = stack_range
        self.gt_scores = gt_scores or {}
        self.slice_num = slice_num
        self.slice_size = slice_size
        self.paired_cases = paired_cases
        self.sampling_method = sampling_method
        if fixed_case2 is not None:
            self.fixed_im2 = nib.load(f'{self.data_root}/{fixed_case2}/ct.nii.gz')
            self.fixed_data2 = np.array(self.fixed_im2.dataobj)
        else:
            self.fixed_im2 = None
            self.fixed_data2 = None
    
    def get_two_labeled_cases(self, case1, case2):

        def sample_slices(num_slices, num_sampled, voxel_size):

            if self.sampling_method is not None:
                sampling_method = self.sampling_method
            
            else:
                a = np.random.random()
                sampling_method='subvolume' if a < 0.5 else 'gap'

            all_indices = np.array(list(range(num_slices)))
            if num_slices < num_sampled:
                return all_indices, voxel_size
            
            if sampling_method == 'gap':
                k = int(np.ceil(num_slices / num_sampled))
                indices = all_indices[::k]
                voxel_size[2] = voxel_size[2] * k
                return indices, voxel_size
            
            if sampling_method == 'subvolume':
                start_i = np.random.randint(0, num_slices - num_sampled + 1)
                indices = all_indices[start_i: start_i + num_sampled]
                return indices, voxel_size

        im1 = nib.load(f'{self.data_root}/{case1}/ct.nii.gz')
        data1 = np.array(im1.dataobj)

        if self.fixed_case2 is not None:
            case2 = self.fixed_case2
            im2 = nib.load(f'{self.data_root}/{case2}/ct.nii.gz')
            data2 = np.array(im2.dataobj)
        else:
            im2 = self.fixed_im2
            data2 = self.fixed_data2

        vox1 = np.abs([im1.affine[0, 0], im1.affine[1, 1], im1.affine[2, 2]])
        vox2 = np.abs([im2.affine[0, 0], im2.affine[1, 1], im2.affine[2, 2]])

        scores1 = np.array(self.gt_scores[case1]['score'])
        scores2 = np.array(self.gt_scores[case2]['score'])

        # sample slices
        indices1, vox1 = sample_slices(data1.shape[2], self.slice_num-2, vox1)
        indices2, vox2 = sample_slices(data2.shape[2], self.slice_num, vox2)

        stack1 = np.zeros((self.slice_size, self.slice_size, self.slice_num))
        stack2 = np.zeros((self.slice_size, self.slice_size, self.slice_num))
        mask1 = np.zeros(self.slice_num)
        mask2 = np.zeros(self.slice_num)

        # starting dumb slice
        stack1[:, :, 0] = 1
        # ending dumb slice
        stack1[:, :, len(indices1) + 1] = -1

        stack1[:, :, 1: len(indices1) + 1] = data1[:, :, indices1]
        stack2[:, :, : len(indices2)] = data2[:, :, indices2]
        mask1[: len(indices1) + 2] = 1
        mask2[: len(indices2)] = 1
        stack_scores1 = scores1[indices1]
        stack_scores2 = scores2[indices2]

        scores1_range = min(stack_scores1), max(stack_scores1)

        index_2in1 = []
        for s in stack_scores2:
            if s < scores1_range[0]:
                index_2in1.append(mask1.sum() - 1)
            elif s > scores1_range[1]:
                index_2in1.append(0)
            else:
                insert_index = np.argmin(np.abs(s - stack_scores1))
                index_2in1.append(insert_index)
        for i in range(len(index_2in1), self.slice_num):
            index_2in1.append(-1)

        truncated_norm_dist = []
        for ind in index_2in1:
            sigma = 5 #mm
            loc = ind * vox1[2]
            x = np.array(list(range(self.slice_num))) * vox1[2]
            if ind < 0:
                truncated_norm_dist.append([-1] * len(x))
            else:
                p = norm.pdf(x, loc=loc, scale=sigma)
                p[mask1 == 0] = 0
                p /= p.sum()
                truncated_norm_dist.append(p)
            
        res = {
            'stack1': stack1.astype(np.float32),
            'stack2': stack2.astype(np.float32),
            'mask1': mask1.astype(np.bool_), 
            'mask2': mask2.astype(np.bool_), 
            # 'scores1': stack_scores1.astype(np.float32),
            # 'scores2': stack_scores2.astype(np.float32),
            'vox1': vox1,
            'vox2': vox2,
            'index_2in1': np.array(index_2in1, dtype=np.int32),
            'index_2in1_gaussian': np.array(truncated_norm_dist, dtype=np.float32),
        }

        return res
    
    def __len__(self):
        return len(self.cases)

    
    def __getitem__(self, index):
        
        case = self.cases[index]
        if case in self.gt_scores:
            if self.fixed_case2:
                case2 = self.fixed_case2
            else:
                index2 = np.random.choice([i for i in range(len(self.cases)) if i != index])
                case2 = self.cases[index2]
            # index2 = 1
            res = self.get_two_labeled_cases(case, case2)
            res['index'] = index
            return res