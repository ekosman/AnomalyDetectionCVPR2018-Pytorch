
import os

import cv2

import argparse
parser = argparse.ArgumentParser(description="vidoe annotatate maker")

parser.add_argument('--path_list', default='[../kinetics2/kinetics2/AnomalyDetection]',
                    help="list of vidoe paths to be annotatated, must be the same length as the normal_or_not and dir_list")

parser.add_argument('--dir_list', default='[[/kinetics2/kinetics2/,Fighting039_x264.mp4]]',
                    help="list of paths to be annotatated. must be the same length as the path_list and normal_or_not")

parser.add_argument('--normal_or_not', default=[True],
                    help="if the video being anotated in normal or abnormal. must be the same length as the path_list and dir_list")

parser.add_argument('--file_name', default="Demo_anmotation",
                    help="the name of the end annotation file")


#this is for the adding of new file for training or etc
#https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch/issues/18
def annotatate_file(path_list,dir_list,normal=[True],file_name="Demo_anmotation"):
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
        if normal[i]==True:
            folder="Normal"
            start="-1"
            end="-1"
        else:
            print("error this is not sorted yet")

        videoReader = cv2.VideoCapture(path)
        length = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))

        str1= dir_list[i][0]+"/"+dir_list[i][1]+" "+str(length)+" "+folder+" "+start+" "+end+" -1 -1"
        file.write(str1)

    file.close()
    home=os.getcwd()
    return home+"/"+file_name+".txt"


if __name__ == '__main__':
    args = parser.parse_args()
    annotatate_file(args.path_list, args.dir_list, normal=args.normal_or_not, file_name=args.file_name)
