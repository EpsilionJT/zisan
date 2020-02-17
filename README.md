# Install zisan
######  JintuZheng  Jan 25th 2020
Last page:  [wiki: Strat pytorch](http://jintupersonal.com/zisan/doc/1.html)

## Step1: Install zisan package
Two easy ways to install zisan.
Input the follow command:

    pip install zisan
   
  Also you can download the wheel file on the official site:
  http://jintupersonal.com/zisan/


Pay attention to the lastest version to modify your pip command.


## Step2: Prepare the weight files for zisan
Firstly you should download the weights files for zisan.
You can find the url button on homepage: Download weights
link：https://pan.baidu.com/s/1qj-Lpe4OKV0L-w9uKO8EFw 
key：x9wl 
The weight files structure is followed:
> Instance_Seg_weight (Folder)
> >Jintu_SEG_Interactive.pth (178 MB)

>ObjectDetect_data_weights (Folder)
>>runBox.zip (475 MB)

Unzip runBox.zip, you will see three folders, they are **"cfgs", "data", "weights"**.

> cfgs: Here put the network configs

> data: Put your data preparing to train 

> weights: Here save the raw yolo weights and your last output weights

## IMPORTANT  !
**DON'T RECHANGE THE FOLDER'S NAME !**

**DON'T RECHANGE THE FOLDER'S NAME !**

**DON'T RECHANGE THE FOLDER'S NAME !**

you can't rechange the "cfgs", "weights", "data" and their child folders name, because zisan. Objedetect class will use the refer all folders' name, if you rechange them, it will happen errors. But you can set your current path and as parameters in function.

more details you can refer the following page: [wiki: Train your dataset](http://jintupersonal.com/zisan/doc/7.html)

How to use these weights?
You will find your answer in the following Demo Courses.

Next page:  [wiki: Class ImgSeg](http://jintupersonal.com/zisan/doc/3.html)

# Demo: A person segmentation
######  JintuZheng  Jan 26th 2020
Last page:  [wiki: Demo: A Box segmentation](http://jintupersonal.com/zisan/doc/4.html)
## Prepare a person picture
![From davis 2017](http://jintupersonal.com/zisan/doc/images/5.1.jpg?v=1&type=image)

(This picture is from dataset davis2017)
Find the person and give him bone marks, like the following. Maybe it's very abstract. Monofilament doesn't affect our segmentation of objects。

![](http://jintupersonal.com/zisan/doc/images/5.2.jpg?v=1&type=image)

## Import Packages

    from zisan.Seg.Interface import ImgSeg, markTools
    import os
    import numpy as np
    import cv2
    from PIL import Image

## Draw mask

    lines=[[(281,120),(267,341)],[(279,157),(208,171)],[(309,170),(308,250)],[(275,233),(370,341)]]
    img=Image.open(current_path+'/temp/1.jpg').convert('RGB')
    markpen=markTools(img.height,img.width)
    for line in lines:
    	markpen.curveDraw(line,is_Pos=True)

[(281,120),(267,341)] is a line element
lines is a line list=[[(,),(,)]]

![mask Preview](http://jintupersonal.com/zisan/doc/images/5.3.jpg?v=1&type=image)


## Instance segmentation

    re=markpen.getMark_result(is_showPreview=True)
    model.ImgSeg_SingleObj(img,re,is_showPreview=True)
   
   Binary mask result preview:
   ![result](http://jintupersonal.com/zisan/doc/images/5.4.jpg?v=1&type=image)


Fade mask show preview:

![fade](http://jintupersonal.com/zisan/doc/images/5.5.jpg?v=1&type=image)

zisan have the Yolov3 interface to train your own dataset.
Please view the next part:
[wiki: Package: Object detect](http://jintupersonal.com/zisan/doc/6.html)

