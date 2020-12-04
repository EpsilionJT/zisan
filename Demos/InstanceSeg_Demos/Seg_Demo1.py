from zisan.Seg.Interface import ImgSeg, markTools
import os
import numpy as np
import cv2
import time


current_path = os.path.dirname(__file__)

Pos_points=[(85,83),(307,257)] #(x,y) wh This is a anchor line which on the object we want
Nav_points=[(98,63),(345,96)] #(x,y) wh  This is a anchor line which on the background

model=ImgSeg(current_path+'/Jintu_SEG_Interactive.pth')

markpen=markTools(291,435)
markpen.curveDraw(Pos_points,is_Pos=True)
#markpen.pointDraw(Pos_points,7,True)
markpen.curveDraw(Nav_points,is_Pos=False)
re=markpen.getMark_result(is_showPreview=False) # Preview the line marks

st=time.time()
model.ImgSeg_SingleObj_FromFile(current_path+'/temp/2.jpg',re,is_showPreview=False)
ed=time.time()
print(ed-st)



