# -*- coding: utf-8 -*-
import torch
import logging

def get_logger(filepath, log_info):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + log_info + '-' * 30)
    return logger

def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)

def NonOverlappingCropPatches(im, patch_size=32, stride=32):
    b, c, w, h = im.shape
    patch_list = []
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = im[:, :, i :i  + patch_size, j :j + patch_size].unsqueeze(1)
            patch_list.append(patch)
    patches = torch.cat(patch_list, dim=1)
    return patches
