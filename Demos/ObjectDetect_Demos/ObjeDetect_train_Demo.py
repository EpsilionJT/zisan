from zisan.ObjDetect.Interface import ObjDetect_train, ObjDetect_Preprocess
import os

'''
Before running train, you must keep the data path in the correct position
Details you can read the Doc: http://jintupersonal.com/zisan/Doc/

'''


if __name__ == "__main__":  
    #current_path = os.path.dirname(__file__)
    current_path = os.getcwd()

    
    pr=ObjDetect_Preprocess(classnames=['RBC'],currentpath=current_path) # current path is needed
    #pr.clear_data() #clear all data 
    
    trainModel=ObjDetect_train(current_path)
    trainModel.Run(cfg='yolov3-tiny.cfg',epochs=10)
    
    