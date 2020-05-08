
import os

import cv2

#this is for the adding of new file for training or etc
#https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch/issues/18
def annotatate_file(path_list,dir_list,normal=True,file_name="Demo_anmotation"):
    #path lenght start end
    #Fighting/Fighting047_x264.mp4 4459 Fighting 200 1830 -1 -1
    #Testing_Normal_Videos_Anomaly/Normal_Videos_872_x264.mp4 530 Normal -1 -1 -1 -1
    if os.path.exists(file_name+".txt")==True:
        os.remove(file_name+".txt")
    file = open(file_name+".txt", "a")
    if type(path_list)!=list():
        path_list=[path_list]
    for i,path in enumerate(path_list):
        print(path)
        assert os.path.exists(path)
        if normal==True:
            folder="Normal"
            start="-1"
            end="-1"
        else:
            print("error this is not sorted yet I will update this when it is working.")
            print("If this is for use with video_demo.py plese set supplyed video at normal=True.")

        videoReader = cv2.VideoCapture(path)
        length = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))

        str1= dir_list[i][0]+"/"+dir_list[i][1]+" "+str(length)+" "+folder+" "+start+" "+end+" -1 -1"
        file.write(str1)

    file.close()
    home=os.getcwd()
    return home+"/"+file_name+".txt"
