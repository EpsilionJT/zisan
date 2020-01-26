from zisan.Seg.Interface import ImgSeg, markTools
import os
import numpy as np
import cv2
from PIL import Image

current_path = os.path.dirname(__file__)

lines=[[(281,120),(267,341)],[(279,157),(208,171)],[(309,170),(308,250)],[(275,233),(370,341)]]
model=ImgSeg(current_path+'/Jintu_SEG_Interactive.pth')

img=Image.open(current_path+'/temp/1.jpg').convert('RGB')
markpen=markTools(img.height,img.width)
for line in lines:
    markpen.curveDraw(line,is_Pos=True)
re=markpen.getMark_result(is_showPreview=True)
model.ImgSeg_SingleObj(img,re,is_showPreview=True)





