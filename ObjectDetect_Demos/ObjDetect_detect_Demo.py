from zisan.ObjDetect.Interface import ObjDetect_detect, ObjDetect_train, ObjDetect_Preprocess
import os
from PIL import Image

'''
Before running detect, you must keep the data path in the correct position
Details you can read the Doc: http://jintupersonal.com/zisan/Doc/

'''

if __name__ == "__main__":  
    current_path = os.path.dirname(__file__)

    detectModel=ObjDetect_detect(cfg='yolov3-tiny.cfg',currentpath=current_path)
    #re=detectModel.detect_fromfiles(images='D:/tstImages',log_print=True) # images is the path where all the test images
    img=Image.open('D:/1.jpg').convert('RGB')
    re=detectModel.detect_from_RGBimg(img,is_showPreview=True)

    for info in re: # get one img classesNums
        print(info[1])
    
    for info in re: # get one img result_boxes
        print(info[0])

    #log_print true means you can read the results in the terminal
    
    #re=[img0,img1,img2]
    #img0=[result_boxes,classesnum]
    #result_boxes=(x1,x2,y1,y2)   
    #classnum=[(nums,class1),(nums,class2)] nums is int classX is str

