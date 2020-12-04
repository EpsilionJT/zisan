from skimage import io
from matplotlib import pyplot as plt
import random
import cv2
import os
import numpy as np

def pngToJpg(src):
    jpgimg=np.zeros([src.shape[0],src.shape[1],3]).astype(np.uint8)#因为plt的图片是RGB而cv的图片是BGR
    jpgimg[:,:,(0,1,2)]=src[:,:,(0,1,2)]#通道转换
    return jpgimg

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def getFiles(dir):
    flist=[]
    for root,dirs,files in os.walk(dir+'/'):
            for file in files:
                flist.append(os.path.join(root,file))
    return flist

def add_roi(backimg,logoimg,logox0,logoy0):
    backimg=backimg.copy()
    logoRs,logoCs=logoimg.shape[0],logoimg.shape[1]#算出logo的空间参数
    for y in range(logoy0,logoy0+logoRs):
            for x in range(logox0,logox0+logoCs):
                backimg[y,x,:]=logoimg[y-logoy0,x-logox0,:]
    return backimg

def scaleTransfrom(tar_img,W,H,x0,x1,y0,y1):  # 对坐标进行放缩
    h,w,_=tar_img.shape
    scale={'x_w':float(w/W),'y_h':float(h/H)} #scale for shrink
    x0 *= scale['x_w']
    x1 *= scale['x_w']
    y0 *= scale['y_h']
    y1 *= scale['y_h']
    return int(x0),int(x1),int(y0),int(y1)

def get_random_files(dir,proportion):
    fl = getFiles(dir)
    rlist=[ i for i in range(0,len(fl))]
    random.shuffle(rlist)
    fslist = []
    for idx in range(0,int(proportion*len(fl))):
        fslist.append(fl[rlist[idx]])
    return fslist

def newMatUC3(width,height,colorR,colorG,colorB):#创建空白图像
    img=np.zeros([height,width,3], np.uint8)
    img[:,:,0]=colorR
    img[:,:,1]=colorG
    img[:,:,2]=colorB
    return img
    
def roi_cutPoint(srcimg,x0,x1,y0,y1):#切割图像
    return srcimg[y0:y1,x0:x1]