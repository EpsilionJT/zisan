#################################################################
### @author：郑晋图 JintuZheng
### Email: jintuzheng@outlook.com
### Remarks: Please respect my labor achievements, and contact me to modify the source code
### Date:2020-01-23
################################################################

import numpy as np
import os
import cv2
from .model import model
from .utils import load_frames,overlay_bin,overlay_fade
from PIL import Image
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

class ImgSeg(object):
    def __init__(self,weight_path):
        self.Inetmodel = model(weight_path) #load the model init need more time

    def Preview(self,img,mask):
        img[:,:,(0,1,2)]=img[:,:,(2,1,0)]#通道转换
        cv2.imshow("Preview_Jintu",img)
        cv2.imshow("Preview_mask_Jintu",mask)
        cv2.waitKey(0)
            
    def rawProcess(self,frame_list,object_mark,is_showPreview):   
        frames = np.stack(frame_list, axis=0)
        self.Inetmodel.loadframes(frames) # reset all frames
        _, height, width = frames.shape[:3]
        self.Inetmodel.Run_interaction(object_mark)
        current_mask = np.zeros((1, height, width), dtype=np.uint8)
        current_mask=self.Inetmodel.Get_mask_index(0)
        result=overlay_bin(frames[0],current_mask)
        if is_showPreview :
            self.Preview(overlay_fade(frames[0],current_mask),result)
        return result,overlay_fade(frames[0],current_mask)

    def ImgSeg_SingleObj_FromFile(self,img_path,object_mark,is_showPreview=False):
        frame_list=[]
        frame_list.append(np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8))
        return self.rawProcess(frame_list,object_mark,is_showPreview)

    def ImgSeg_SingleObj(self,img_RGB,object_mark,is_showPreview=False): #Image.open(img_path).convert('RGB')
        frame_list=[]
        frame_list.append(np.array(img_RGB, dtype=np.uint8))
        return self.rawProcess(frame_list,object_mark,is_showPreview)
    

class markTools(object):
    def __init__(self,height,width):
        self.canvas=np.zeros((height,width), np.uint8) #create a new canvas
        self.height=height
        self.width=width
        self.thickness=7
        self.lineType=4
        self.color=255
        self.pen=[[5,1],[6,2],[7,3],[8,4],[9,5],[10,5],[15,5]] # [thickness point_size]

    def pointStrength(self,level=4): #max level is 7
        thickness,point_size=self.pen[level-1]
        return thickness,point_size
    
    def checkColor(self,flag):
        if flag==False:
            self.color=100
        else: 
            self.color=255

    def curveDraw(self,points=[(0,0)],thickness=10,is_Pos=True):
        self.checkColor(is_Pos)
        if len(points)<2:
            return
        ptStart=points[0]
        ptEnd=points[1]
        for index in range(1,len(points)):
            ptEnd=points[index]
            self.canvas=cv2.line(self.canvas, ptStart, ptEnd, self.color, thickness, self.lineType)
            ptStart=ptEnd

    def pointDraw(self,points=[(0,0)],point_sizeLevel=4,is_Pos=True):
        self.checkColor(is_Pos)
        tk,psz=self.pointStrength(point_sizeLevel)
        for cur_point in points:
            cv2.circle(self.canvas,cur_point,psz,self.color,tk)
            
    def getMark_result(self,is_showPreview=False):
        if is_showPreview:
            cv2.imshow('markPreview',self.canvas)
            cv2.waitKey(0)
        #处理顺序的逻辑不能错
        self.canvas[self.canvas==255]=1 #Pos
        self.canvas[self.canvas==0]=-1 #background
        self.canvas[self.canvas==100]=0 # Na
        return self.canvas
    def getMark_result_from_gray(img_gray2):
        img_gray=img_gray2.copy()
        img_gray[img_gray==255]=1 #Pos
        img_gray[img_gray==0]=-1 #background
        img_gray[img_gray==100]=0 # Na
        return img_gray

    def mergeMask_fade_from_gray(image,mask_gray):
        from scipy.ndimage.morphology import binary_erosion, binary_dilation
        mask_gray[np.where(mask_gray==255)]=1
        mask=mask_gray
        im_overlay = image.copy()
        # Overlay color on  binary mask
        binary_mask = mask == 1
        not_mask = mask != 1
        # Compose image
        im_overlay[not_mask] = 0.4 * im_overlay[not_mask]
        countours = binary_dilation(binary_mask) ^ binary_mask
        im_overlay[countours,0] = 0
        im_overlay[countours,1] = 255
        im_overlay[countours,2] = 255
        return im_overlay.astype(image.dtype)

