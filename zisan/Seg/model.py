from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torchvision.transforms as transforms

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import os
import argparse
import copy
import random

# my libs
from .utils import ToCudaVariable, ToCudaPN, load_UnDP, Get_weight, overlay_davis, overlay_checker, overlay_color, overlay_fade
from .interaction_net import Inet


# davis
from .davisinteractive.utils.scribbles import scribbles2mask, annotated_frames

import warnings
warnings.filterwarnings('ignore')
#current_path = os.path.dirname(__file__)


class model():
    def __init__(self,weight_path):
        self.model_I = Inet()
        if torch.cuda.is_available():
            #print('init GPU')
            self.model_I = nn.DataParallel(self.model_I)
            self.model_I.cuda()
            self.model_I.load_state_dict(torch.load(weight_path))
        else:
            #print('init CPU')
            self.model_I.load_state_dict(load_UnDP(weight_path))
        self.model_I.eval()

    def loadframes(self,frames):
        self.frames = frames
        self.num_frames, self.height, self.width = self.frames.shape[:3]
        self.init_variables(self.frames)

    def init_variables(self, frames):
        self.all_F = torch.unsqueeze(torch.from_numpy(np.transpose(frames, (3, 0, 1, 2))).float() / 255., dim=0) # 1,3,t,h,w
        self.all_E = torch.zeros(1, self.num_frames, self.height, self.width)  # 1,t,h,w
        self.prev_E = torch.zeros(1, self.num_frames, self.height, self.width)  # 1,t,h,w
        self.dummy_M = torch.zeros(1, self.height, self.width).long()
        # to cuda
        self.all_F, self.all_E, self.prev_E, self.dummy_M = ToCudaVariable([self.all_F, self.all_E, self.prev_E, self.dummy_M], volatile=True)
        
        self.ref = None
        self.a_ref = None
        self.next_a_ref = None
        self.prev_targets = []


    def Run_interaction(self, scribble_mask): # scribble_mask is a tensor [Height,width]  1 means object 0 means background
        target=0
        self.tar_P, self.tar_N = ToCudaPN(scribble_mask)
        self.all_E[:,target], _, self.ref = self.model_I(self.all_F[:,:,target], self.all_E[:,target], self.tar_P, self.tar_N, self.dummy_M, [1,0,0,0,0]) # [batch, 256,512,2]
        
        
    def Get_mask(self):
        return torch.round(self.all_E[0]).data.cpu().numpy().astype(np.uint8) 

    def Get_mask_range(self, start, end):
        pred_masks = torch.round(self.all_E[0, start:end]).data.cpu().numpy().astype(np.uint8) # t,h,w
        return torch.round(self.all_E[0, start:end]).data.cpu().numpy().astype(np.uint8)

    def Get_mask_index(self, index):
        return torch.round(self.all_E[0, index]).data.cpu().numpy().astype(np.uint8)




