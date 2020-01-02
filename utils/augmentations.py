from albumentations import ( Compose, RandomRotate90, Flip, Transpose, GridDistortion, ElasticTransform)
import numpy as np
import cv2
import torch

"""
In some systems, in the multiple GPU regime PyTorch may deadlock the DataLoader if OpenCV was compiled with OpenCL optimizations. Adding the following two lines before the library import may help. For more details https://github.com/pytorch/pytorch/issues/1355
"""
if torch.cuda.device_count() > 1:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

def adni_qnat_aug(prob=0.5, torch_tensor=None, additional_targets={}):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(), 
        GridDistortion(interpolation=0),
        ElasticTransform(interpolation=0)
    ], p=prob, additional_targets=additional_targets)
