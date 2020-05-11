
from network.c3d import C3D #network.
from data_loader import VideoIterTrain
import torch

from utils.utils import set_logger, build_transforms

import os

from feature_extractor import FeaturesWriter
from tqdm import tqdm


from network.anomaly_detector_model import AnomalyDetector, RegularizedLoss, custom_objective
from network.model import static_model

from features_loader import FeaturesLoaderVal


import numpy as np

import cv2
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description="Video demo maker")

parser.add_argument('--video_parth_list', default=["/media/peter/Maxtor/UCF_Crimes/Videos/Fighting/Fighting039_x264.mp4","/media/peter/Maxtor/UCF_Crimes/Videos/Fighting/Fighting040_x264.mp4"],
                    help="list of videos to be used for demo")
parser.add_argument('--features_dir', default="./demo_video",
                    help="path to dir where you want the featuers saved")
parser.add_argument('--model_dir', default='/home/peter/Documents/actionID/AnomalyDetectionCVPR2018-Pytorch-master/short_60_low_mem/exps/model',
                    help="path to the tarined AD model")


args = parser.parse_args()

video_parth_list = args.video_parth_list
features_dir = args.features_dir
model_dir = args.model_dir



def figure2opencv(figure):
    figure.canvas.draw()
    img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def cd3_extartion(video_parth,device=None):

    batch_size=1
    train_frame_interval=2
    clip_length=16


    single_load=True #should not matter
    home=os.getcwd()
    pretrained_3d=home+"/c3d-pytorch-master/c3d.pickle"

    if device==None:
        device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")


    #Load clips
    print("doing train loader")
    train_loader = VideoIterTrain(dataset_path=None,
                                  annotation_path=video_parth,
                                  clip_length=clip_length,
                                  frame_stride=train_frame_interval,
                                  video_transform=build_transforms(),
                                  name='train',
                                  return_item_subpath=False,
                                  single_load=single_load)
    print("train loader done, train_iter now")
    train_iter = torch.utils.data.DataLoader(train_loader,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=32,  # 4, # change this part accordingly
                                             pin_memory=True)

    #Possesing with CD3
    print("Now loading the data to C3D netowr")
    network = C3D(pretrained=pretrained_3d)
    network.to(device)

    if not os.path.exists(features_dir):
        os.mkdir(features_dir)

    features_writer = FeaturesWriter()

    dir_list=[]

    for i_batch, (data, target, sampled_idx, dirs, vid_names) in tqdm(enumerate(train_iter)):
        with torch.no_grad():
            outputs = network(data.cuda())

            for i, (dir, vid_name, start_frame) in enumerate(zip(dirs, vid_names, sampled_idx.cpu().numpy())):
                dir_list.append([dir,vid_name])
                dir = os.path.join(features_dir, dir)
                features_writer.write(feature=outputs[i], video_name=vid_name, start_frame=start_frame, dir=dir)

    features_writer.dump()
    return dir_list

def AD_perdiction(model_dir,dir_list,device=None):

    if device==None:
        device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    #pediction of AD with pertrain network
    network = AnomalyDetector()
    network.to(device)
    net = static_model(net=network,
                           criterion=RegularizedLoss(network, custom_objective).cuda(),
                           model_prefix=model_dir,
                           )
    model_path = net.get_checkpoint_path(20000)
    net.load_checkpoint(pretrain_path=model_path, epoch=20000)
    net.net.to(device)

    from annotation_methods import annotatate_file
    annotation_path=annotatate_file(video_parth,dir_list,normal=True,file_name="Demo_anmotation")

    #runing vedio in to network
    data_loader = FeaturesLoaderVal(features_path=features_dir,
                                    annotation_path=annotation_path)

    data_iter = torch.utils.data.DataLoader(data_loader,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,  # 4, # change this part accordingly
                                            pin_memory=True)
    print("it is over")

    for features, start_end_couples, feature_subpaths, lengths in tqdm(data_iter):
        # features is a batch where each item is a tensor of 32 4096D features
        features = features.to(device)
        with torch.no_grad():
            input_var = torch.autograd.Variable(features)
            outputs = net.predict(input_var)[0]  # (batch_size, 32)
            outputs = outputs.reshape(outputs.shape[0], 32)
            for vid_len, couples, output in zip(lengths, start_end_couples, outputs.cpu().numpy()):
                y_true = np.zeros(vid_len)
                segments_len = vid_len // 32
                for couple in couples:
                    if couple[0] != -1:
                        y_true[couple[0]: couple[1]] = 1
                y_pred = np.zeros(vid_len)
                print()
                for i in range(32):
                    segment_start_frame = i * segments_len
                    segment_end_frame = (i + 1) * segments_len
                    y_pred[segment_start_frame: segment_end_frame] = output[i]

    print(y_true)
    print(y_pred)
    print("it is over")
    return y_pred


def GUI(video_parth,y_pred,save="Just save",s_path="output_video"):
    DISPLAY_IMAGE_SIZE = 500
    BORDER_SIZE = 50#100
    FIGHT_BORDER_COLOR = (0, 0, 255)
    NO_FIGHT_BORDER_COLOR = (0, 255, 0)

    plot_range = 100
    # violenceDetector = ViolenceDetector()
    videoReader = cv2.VideoCapture(video_parth)
    isCurrentFrameValid, currentImage = videoReader.read()

    if save == True or save == "Just save":
        #fps = videoReader.get(cv2.CV_CAP_PROP_FPS)
        fps = videoReader.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(s_path+'.avi', fourcc,fps,(600, 300))

    #farme_window=[0]*clip_length
    fig = plt.figure()
    farme_cout = 0
    length = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))
    while isCurrentFrameValid:
        farme_cout = farme_cout + 1

        targetSize = DISPLAY_IMAGE_SIZE - 2 * BORDER_SIZE
        currentImage = cv2.resize(currentImage, (targetSize, targetSize))

        NO_FIGHT_BORDER_COLOR = NO_FIGHT_BORDER_COLOR
        resultImage = cv2.copyMakeBorder(currentImage,
                                         BORDER_SIZE,
                                         BORDER_SIZE,
                                         BORDER_SIZE,
                                         BORDER_SIZE,
                                         cv2.BORDER_CONSTANT,
                                         value=NO_FIGHT_BORDER_COLOR)

        resultImage = cv2.resize(resultImage, (300, 300))

        #fig = plt.figure()
        # print(pec_store)
        #print(farme_cout)
        #print(y_pred[farme_cout-1])
        plt.plot(farme_cout, y_pred[farme_cout-1], color='green', marker='o', linestyle='-', linewidth=2, markersize=2)
        # plt.plot(it_uesed, pec_store2, c="r")
        # plt.xlim()
        plt.ylim(0, 1.0)
        if farme_cout<100:
            plt.xlim(0, 100)
        else:
            plt.xlim(farme_cout-100, farme_cout+100)

        plt.xlim(0, length)
        plot_img = figure2opencv(fig)
        plot_img = cv2.resize(plot_img, (300, 300))  # (0, 0), None, .25, .25)

        resultImage = np.concatenate((resultImage, plot_img), axis=1)


        if save == True or save=="Just save":
            # Write the frame into the file 'output.avi'
            print("saving")
            out.write(resultImage)

        if save!="Just save":
            cv2.imshow("Violence Detection", resultImage)
        else:
            print(str(farme_cout)+"/"+str(length))

        userResponse = cv2.waitKey(1)
        if userResponse == ord('q'):
            videoReader.release()
            cv2.destroyAllWindows()
            break

        else:
            isCurrentFrameValid, currentImage = videoReader.read()


    if save == True or save == "Just save":
        videoReader.release()
        out.release()

    cv2.destroyAllWindows()

    #results and video play at the same time

if __name__ == '__main__':
    home=os.getcwd()

    args = parser.parse_args()

    video_parth_list=args.video_parth_list
    features_dir=args.features_dir
    model_dir=args.model_dir

    # # make video file format
    # if "peter" in home:
    #     video_parth = "/media/peter/Maxtor/UCF_Crimes/Videos/Fighting/Fighting047_x264.mp4"
    # else:
    #     video_parth = "/home/barbara/Desktop/Fighting047_x264.mp4"


    # from os import listdir
    # from os.path import isfile, join
    # mypath="/media/peter/Maxtor/UCF_Crimes/Videos/Fighting"
    # #video_parth_list = [mypath+"/"+f for f in listdir(mypath) if isfile(join(mypath, f))]
    # #print("lenght "+str(len(video_parth_list)))
    #video_parth_list=["/media/peter/Maxtor/UCF_Crimes/Videos/Fighting/Fighting039_x264.mp4"]

    #features_dir = r"./demo_video"

    # defining pertrained modle of AD
    #model_dir = r'/home/peter/Documents/actionID/AnomalyDetectionCVPR2018-Pytorch-master/short_60_low_mem/exps/model'
    # "/home/peter/Documents/actionID/AnomalyDetectionCVPR2018-Pytorch-master/AnomalyDetectionCVPR2018-Pytorch-master/short_60_low_mem/exps/model"

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    for video_parth in video_parth_list:
        dir_list=cd3_extartion(video_parth,device=device)
        y_pred=AD_perdiction(model_dir, dir_list, device=device)

        import pickle
        print(home+features_dir[1:]+dir_list[0][0]+"/"+dir_list[0][1]+'_y_pred.pkl')
        with open(home+features_dir[1:]+"/"+dir_list[0][0]+"/"+dir_list[0][1]+'_y_pred.pkl', 'wb') as f:
            pickle.dump(y_pred, f)
        #with open(features_dir+dir_list[0][0]+"/"+dir_list[0][1]+'_y_pred.pkl', 'rb') as f:
        #    y_pred = pickle.load(f)

        GUI(video_parth, y_pred,save="Just save",s_path=features_dir+dir_list[0][0]+"/"+dir_list[0][1]+"_demoe")
