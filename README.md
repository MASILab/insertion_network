# Insertion Network

**Insertion Network** is a method for building slice-level correspondence between two axial body CT image sequences.  
Given a *test* CT volume and a *template* CT volume, the model establishes correspondence by **inserting slices from the template image into their correct locations within the test image sequence**.

---

## Overview

- **Task**: Slice-level correspondence between two axial CT volumes  
- **Input**:  
  - Test CT image (NIfTI format)  
  - Template CT image (NIfTI format)  
- **Output**: Predicted insertion locations of template slices within the test sequence  
- **Core idea**: Correspondence is built by predicting where each template slice should be inserted into the ordered slice sequence of the test image.

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Inference

To run inference and compute slice correspondence between two CT volumes:

```bash
python inference_args.py \
  --test_nii <test image path> \
  --template_nii <query image path>
```

The output is a visualization showing the predicted insertion positions of the template slices within the test sequence.

### Example Output (query image referes to the template image)

<!-- Replace the image below with your actual output figure -->

<img width="2723" height="738" alt="output" src="https://github.com/user-attachments/assets/5843a644-9878-42d4-92f1-f5c720f2e96f" />


---

## Training

The training code is provided at:

```
exp/Exp_body_ct_semi_supervised_1.0_kl_sinPE_1/train.py
```

This implementation can be used as a **reference** for training a new model on your own dataset.

### Notes on Training

- The provided training setup is **supervised**.
- **Slice-level ground-truth correspondence is required**.
- Please refer to the paper for details on how the ground-truth correspondences are constructed.

---

## Citation

If you use this code or the Insertion Network method in your work, please cite:

```bibtex
@inproceedings{su2026insertion,
  title={Insertion Network for Image Sequence Correspondence Building},
  author={Su, Dingjie and Hong, Weixiang and Dawant, Benoit M. and Landman, Bennett A.},
  booktitle={SPIE Medical Imaging: Image Processing},
  year={2026}
}
```
