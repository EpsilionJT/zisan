# Install zisan
######  JintuZheng  Jan 25th 2020
Last page:  [wiki: Strat pytorch](http://jintupersonal.com/zisan/doc/1.html)

## Step1: Install zisan package
Two easy ways to install zisan.
Input the follow command:

    pip install zisan
   
  Also you can download the wheel file on the official site:
  http://jintupersonal.com/zisan/

Then install locally

    pip install zisan-1.0.9-py3-none-any.whl
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
