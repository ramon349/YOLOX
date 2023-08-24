#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# Copyright (c) Francisco Massa.
# Copyright (c) Ellis Brown, Max deGroot.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import os.path
import pickle
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from yolox.evaluators.voc_eval import voc_eval

from .datasets_wrapper import CacheDataset, cache_read_img
from .voc_classes import VOC_CLASSES
from .voc import VOCDetection 

import pandas as pd 
import nibabel as nib 
import  pdb 

"""
side note information: 
0 is a valid vlaue for the primary class. Not sure where i thought 0 should be background 

"""
class CTAnnotationTransform(object): 
    def __init__(self) -> None:
        #should the first class be zero 
        self.class_to_ind =  dict(zip(CTVOC_CLASSES,range(len(CTVOC_CLASSES)))  ) # i could just write {'kidney':0} but i wont.... 
    def __call__(self,target:pd.Series): 
        """
        Arguments: 
            target (annotation|Pandas Series): the target annotation to be  used 
        returns: 
            a list containing list of bounding boxes [bbox coords , class_names] 
        """
        res = np.empty((0,5)) 
        KIDNEY_LABEL=0 # hardcoded value as we only have 1 label 
        #x is columsn  y is rows 
        bbox =  [ target[f'bbox-{e}'] for e in [1,2,4,5]] #should be in xmin,ymin,xmax,ymax 
        bbox.append(KIDNEY_LABEL) 
        img_height = abs(target['height'])
        img_width = abs(target['width']) 
        res = np.vstack((res,bbox))
        return res,(img_height,img_width) 



CTVOC_CLASSES = ( 
    "kidney"
)
class CTVOCDetection(VOCDetection):
    """
    CT based VOC detection dataset object 
    input is image, target is annotation 

    Args: 
    file_dir: file path to csv specifiyign dataset 
    image_set (string): split to use for dataset (eg. Train,Val,Test)
    target_transform: transformation to perform on the target annotation ( Is also custom)
    dataset_name: which dataset to load. will be CTVOC  

    """
    def __init__(self, data_path, image_set ='train',img_size=(416,416),preproc=None, 
    target_transform = CTAnnotationTransform(), dataset_name =None,cache=False,cache_type='ram',
    vmin=-100,vmax=300):
        self.data = pd.read_csv(data_path)
        self.root = os.path.split(data_path) #TODO HERE BE DRAGONS
        self.data = self.data[self.data['split']==image_set]
        self.image_set = image_set
        self.img_size = img_size
        self.preproc = preproc
        self.vmin= vmin
        self.vmax = vmax
        self.target_transform = target_transform
        self.name = dataset_name
        #no need to specify these as the 
        self._classes = CTVOC_CLASSES
        self.cats = [
            {"id": idx, "name": val} for idx, val in enumerate(CTVOC_CLASSES)
        ]
        self.class_ids = list(range(len(CTVOC_CLASSES)))
        self.ids = list()
        self.num_imgs = self.data.shape[0]
        self.cache = cache
        self.cache_type = cache_type
        self.annotations = self._load_coco_annotations()  
        super(VOCDetection,self).__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir = self.root,
            cache_dir_name=f"cahce_{dataset_name}",
            path_filename= self.data['image'].unique(),
            cache=cache,
            cache_type=cache_type
        )
    def _load_ctcoco_annotations(self): 
        return [self.load_anno_from_idx(_idx) for _idx in range(self.num_imgs)] 
    def load_anno_from_ids(self, index): 
        row = self.data.iloc[index] 
        res,img_info= self.target_transform(row)
        height , width = img_info
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r # TODO: understand what this does 
        resized_info = (int(height * r), int(width * r))
        return  (res,img_info,resized_info)
    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img

    def load_image(self, index):
        from PIL import Image 
        import numpy as np 
        index = int(index) # for some reason i get tensors as indexes 
        img_path = self.data.iloc[index]['png_path']  
        with Image.open(img_path,'r') as f: 
            img = np.array(f)

        return img 
    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)
    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        target, img_info, _ = self.annotations[index]
        img = self.read_img(index)

        return img, target, img_info, index


   
if __name__=="__main__":
    ds = CTVOCDetection("/media/Datacenter_storage/ramon_dataset_curations/kidney_radiomics_yolo/datasets/stu_bbox_dataset.csv",image_set='train')