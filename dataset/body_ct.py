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

    
class DualStackSamplerFromSingleVolume(Dataset):

    def __init__(self, data_root, cases, paired_cases=None, sampling_method=None,
                 stack_range=(0.3, 1), slice_num=256, slice_size=256, gt_scores=None, fixed_case2=None):
        self.data_root = data_root
        self.cases = cases
        self.stack_range = stack_range
        self.gt_scores = gt_scores or {}
        self.slice_num = slice_num
        self.slice_size = slice_size
        self.paired_cases = paired_cases
        self.sampling_method = sampling_method
        self.fixed_case2 = fixed_case2
    
    def random_z_spacing_numpy(
        vol,            # (B, C, Z, Y, X) torch tensor
        affine,         # (4, 4) numpy array or torch tensor
        z_range=(1.0, 3.0)
    ):

        sz = affine[2, 2]
        new_sz = np.random.uniform(*z_range)

        z = vol.shape[2]
        new_z = int(z * sz / new_sz)

        float_indice = np.linspace(0, z - 1, new_z)

        l = float_indice.floor()
        r = float_indice.ceil()

        l_vol = vol[:, :, l]
        r_vol = vol[:, :, r]

        new_vol = l_vol * (float_indice - l) + r_vol * (1 - (float_indice - l))

        new_affine = affine.copy()
        new_affine[2, 2] = new_sz

        return new_vol, new_affine
    
    def get_two_labeled_cases(self, im1, im2, scores1, scores2):

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

        data1 = np.array(im1.dataobj)
        data2 = np.array(im2.dataobj)

        vox1 = np.abs([im1.affine[0, 0], im1.affine[1, 1], im1.affine[2, 2]])
        vox2 = np.abs([im2.affine[0, 0], im2.affine[1, 1], im2.affine[2, 2]])

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
        img = nib.load(f'{self.data_root}/{case}.nii')
        vol, affine = img.get_fdata().astype(np.float32), img.affine
        
        img1, scores1 = self.random_z_spacing(vol, affine)
        img2, scores2 = self.random_z_spacing(vol, affine)

        # print(scores1)
        # print(scores2)
        
        res = self.get_two_labeled_cases(img1, img2, scores1, scores2)

        return res


    def random_z_spacing(self, vol, affine, z_range=(1.0, 5.0)):

        sz = affine[2, 2]
        new_sz = np.random.uniform(*z_range)
        # new_sz = 2.5

        z = vol.shape[2]
        new_z = int(z * sz / new_sz)

        float_indice = np.linspace(0, z - 1, new_z)

        l = np.floor(float_indice).astype(np.int32)
        r = np.ceil(float_indice).astype(np.int32)

        l_vol = vol[:, :, l]
        r_vol = vol[:, :, r]
        

        new_vol = l_vol * (1 - (float_indice - l)) + r_vol * (float_indice - l)

        new_affine = affine.copy()
        new_affine[2, 2] = new_sz

        start_offeset = np.random.randint(0, new_vol.shape[2] // 3)
        end_offeset = np.random.randint(1, new_vol.shape[2] // 3)

        scores = np.linspace(0, vol.shape[2], new_vol.shape[2])

        new_vol = new_vol[:, :, start_offeset: -end_offeset]
        scores = scores[start_offeset: -end_offeset]

        return nib.Nifti1Image(new_vol, new_affine), -scores

class DualStackSamplerFromSingleVolumeTestPhase(DualStackSamplerFromSingleVolume):

    def __getitem__(self, index):
        
        case = self.cases[index]
        img = nib.load(f'{self.data_root}/{case}.nii')
        img2 = nib.load(f'{self.data_root}/template.nii')
        vol, affine = img.get_fdata().astype(np.float32), img.affine
        vol2, affine2 = img2.get_fdata().astype(np.float32), img2.affine
        
        img1, scores1 = self.random_z_spacing(vol, affine)
        img2, scores2 = img2, np.array(list(range(img2.shape[2])))

        # print(scores1)
        # print(scores2)
        
        res = self.get_two_labeled_cases(img1, img2, scores1, scores2)

        return res
    
class DualStackSamplerSemiSupervised(DualStackSamplerFromSingleVolume):


    def __getitem__(self, index):
        
        case = self.cases[index]
        if case in self.gt_scores:
            if self.fixed_case2 is not None:
                another_case = self.fixed_case2
            else:
                other_cases = [k for k in self.cases if k != case]
                another_case = np.random.choice(other_cases)

            img1 = nib.load(f'{self.data_root}/{case}.nii')
            scores1 = np.array(self.gt_scores[case]['score'])
            img2 = nib.load(f'{self.data_root}/{another_case}.nii')
            scores2 = np.array(self.gt_scores[another_case]['score'])

            res = self.get_two_labeled_cases(img1, img2, scores1, scores2)
        
        else:
            a = 1/0
            img = nib.load(f'{self.data_root}/{case}.nii')
            vol, affine = img.get_fdata().astype(np.float32), img.affine
            
            if np.random.rand() < 0.8:
                img1, scores1 = self.random_z_spacing(vol, affine)
                img2, scores2 = self.random_z_spacing(vol, affine)

            else:
                img1, scores1 = self.random_z_spacing(vol, affine)
                img2, scores2 = img, np.array(list(range(img.shape[2])))

            res = self.get_two_labeled_cases(img1, img2, scores1, scores2)

        res['index'] = index
        res['index2'] = self.cases.index(another_case)
        return res



