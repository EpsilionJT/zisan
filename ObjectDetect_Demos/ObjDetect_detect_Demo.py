from zisan.ObjDetect.Interface import ObjDetect_detect, ObjDetect_train, ObjDetect_Preprocess
import os
from PIL import Image

'''
Before running detect, you must keep the data path in the correct position
Details you can read the Doc: http://jintupersonal.com/zisan/Doc/

'''

if __name__ == "__main__":  
    current_path = os.path.dirname(__file__)

    detectModel=ObjDetect_detect(cfg='yolov3-tiny.cfg',currentpath='D:/xxx/runBox')
    img=Image.open('D:/1.jpg').convert('RGB')
    re=detectModel.detect_from_RGBimg(img,is_showPreview=True)
    print(re)
