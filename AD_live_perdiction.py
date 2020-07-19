import sys
# insert at 1, 0 is the script path (or '' in REPL)
import os
home=os.getcwd()
sys.path.insert(0, home[0:-len('GUI_stuff')])

from data_loader import VideoIterTrain
import torch

from utils.utils import set_logger, build_transforms

import pickle

from feature_extractor import FeaturesWriter
from tqdm import tqdm

from network.anomaly_detector_model import AnomalyDetector, RegularizedLoss, custom_objective
from network.model import static_model

from features_loader import FeaturesLoaderVal


import numpy as np

from torch.autograd import Variable

if "/home/barbara/" in os.getcwd():
    pretrained_3d="/home/barbara/Documents/work_git/actionID/AnomalyDetectionCVPR2018-Pytorch-master/c3d-pytorch-master/c3d.pickle"
elif "/home/peter/" in os.getcwd():
    pretrained_3d="/home/peter/Documents/actionID/AnomalyDetectionCVPR2018-Pytorch-master/c3d-pytorch-master/c3d.pickle"


AD_pertrained_model_dir='/home/peter/Documents/actionID/AnomalyDetectionCVPR2018-Pytorch-master/short_60_low_mem/exps/model'


from network.c3d import C3D


from skimage.transform import resize
def get_clip(clip_list, verbose=True):
    """
    Loads a clip to be fed to C3D for classification.
    TODO: should I remove mean here?

    Parameters
    ----------
    clip_name: str
        the name of the clip (subfolder in 'data').
    verbose: bool
        if True, shows the unrolled clip (default is True).

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """

    clip_list = np.array([frame for frame in clip_list])

    #clip = sorted(glob(join('data', clip_name, '*.png')))
    #test = io.imread(clip[0])
    #rtest = resize(test, output_shape=(112, 200), preserve_range=True)
    clip = np.array([resize(frame, output_shape=(112, 200), preserve_range=True) for frame in clip_list])
    clip = clip[:, :, 44:44 + 112, :]  # crop centrally

    # if verbose:
    #     clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
    #     io.imshow(clip_img.astype(np.uint8))
    #     io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)

def cd3_sigle_extartion(input_clips_f,c3d_network=None):
    #input_clips = get_clip(input_clips_f, verbose=True)
    input_clips_t=np.array(input_clips_f)
    input_clips=torch.from_numpy(np.array(input_clips_f))
    transforms=build_transforms()
    input_clips =transforms(input_clips)

    random_seed = 1
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    if c3d_network==None:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
        print("Now doing the C3D network")
        #c3d_network = C3D(pretrained=pretrained_3d)
        #c3d_network.to(device)

        #from C3D_model import C3D
        #from network.model import static_model
        #from network.anomaly_detector_model import AnomalyDetector, RegularizedLoss, custom_objective

        c3d_network = C3D()
        home = os.getcwd()
        os.chdir(home[0:-len('GUI_stuff')])
        c3d_network.load_state_dict(torch.load('c3d.pickle'))
        os.chdir(home)
        c3d_network.cuda()
        c3d_network.eval()


    #X = Variable(input_clips)
    #X = X.cuda()
    X=input_clips
    X=X.unsqueeze(0)
    #X=X.cuda()
    #input_clip=torch.from_numpy(input_clip)

    with torch.no_grad():
        c3d_outputs = c3d_network(X.cuda())

        features_writer = FeaturesWriter()
        start_frame=0
        vid_name='test_sigle_run_output'
        #c3d_outputs[0] as this is for a single use meaning no need to loop over the results
        features_writer.write(feature=c3d_outputs[0], video_name=vid_name, start_frame=start_frame, dir="test")

        avg_segments=features_writer.dump_NO_save()#dump()

    if c3d_network == None:
        return c3d_outputs,avg_segments, device,c3d_network
    else:
        return c3d_outputs,avg_segments

def cd3_extartion(video_parth,device=None,features_dir="./demo_video",c3d_network=None,train_frame_interval=2,clip_length=16):

    batch_size=1
    #train_frame_interval=2
    #clip_length=16

    single_load=True #should not matter
    home=os.getcwd()
    load_home=home[0:-len('GUI_stuff')]
    pretrained_3d=load_home+"/c3d-pytorch-master/c3d.pickle"

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

    if c3d_network==None:
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
            #print("dir " + str(dirs))
            for i, (dir, vid_name, start_frame) in enumerate(zip(dirs, vid_names, sampled_idx.cpu().numpy())):
                dir_list.append([dir,vid_name]) #added by Peter 9/5
                #print("dir "+str(dir))
                dir = os.path.join(features_dir, dir)
                features_writer.write(feature=outputs[i], video_name=vid_name, start_frame=start_frame, dir=dir)
    #print("dumping?")
    features_writer.dump()
    if c3d_network==None:
        return dir_list,network,data
    else:
        return dir_list

def AD_sigle_perdiction(model_dir,c3d_features,lengths=16,device=None,network=None):
    if device==None:
        device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    if network==None:
        print("staring the sigle AD networl")
        # pediction of AD with pertrain network
        network = AnomalyDetector()
        network.to(device)
        net = static_model(net=network,
                           criterion=RegularizedLoss(network, custom_objective).cuda(),
                           model_prefix=model_dir,
                           )
        model_path = net.get_checkpoint_path(20000)
        net.load_checkpoint(pretrain_path=model_path, epoch=20000)
        net.net.to(device)
    else:
        net=network

    #no need for anatation or batch loading of the C3D featuers

    #from annotation_methods import annotatate_file
    #annotation_path=annotatate_file(video_parth,dir_list,normal=True,file_name="Demo_anmotation")

    # #runing vedio in to network
    # data_loader = FeaturesLoaderVal(features_path=features_dir,
    #                                 annotation_path=annotation_path)
    #
    # data_iter = torch.utils.data.DataLoader(data_loader,
    #                                         batch_size=1,
    #                                         shuffle=False,
    #                                         num_workers=1,  # 4, # change this part accordingly
    #                                         pin_memory=True)
    #print("loading of data done")

    #for features, start_end_couples, feature_subpaths, lengths in tqdm(data_iter):
        # features is a batch where each item is a tensor of 32 4096D features
    c3d_features=torch.from_numpy(c3d_features)
    #print(c3d_features.shape)
    features = c3d_features.to(device)
    with torch.no_grad():
        input_var = torch.autograd.Variable(features)
        outputs = net.predict(input_var)[0]  # (batch_size, 32)
        outputs = outputs.reshape(1, 32)#outputs.shape[0]
        for vid_len,  output in zip([lengths],  outputs.cpu().numpy()):
            y_true = np.zeros(vid_len)
            segments_len = vid_len // 32
        #         for couple in couples:
        #             if couple[0] != -1:
        #                 y_true[couple[0]: couple[1]] = 1
            y_pred = np.zeros(vid_len)
            for i in range(32):
                segment_start_frame = i * segments_len
                segment_end_frame = (i + 1) * segments_len
                y_pred[segment_start_frame: segment_end_frame] = output[i]

    #print(y_true)
    #print(y_pred)
    #print("it is over")
    return y_pred


def AD_perdiction(model_dir,dir_list,features_dir,video_parth,device=None):

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
    print("loading of data done")

    for features, start_end_couples, feature_subpaths, lengths in tqdm(data_iter):
        # features is a batch where each item is a tensor of 32 4096D features
        print(features.shape)
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
    #print("it is over")
    return y_pred

def network_setup(ad_model_dir='/home/peter/Documents/actionID/AnomalyDetectionCVPR2018-Pytorch-master/short_60_low_mem/exps/model'):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    c3d_network = C3D(pretrained=pretrained_3d)
    c3d_network.to(device)

    print("staring the sigle AD networl")
    # pediction of AD with pertrain network
    AD_network = AnomalyDetector()
    AD_network.to(device)
    net = static_model(net=AD_network,
                       criterion=RegularizedLoss(AD_network, custom_objective).cuda(),
                       model_prefix=ad_model_dir,
                       )
    model_path = net.get_checkpoint_path(20000)
    net.load_checkpoint(pretrain_path=model_path, epoch=20000)
    net.net.to(device)

    return device,c3d_network,net

def testing(clip_size=16,video_input=0):
    import cv2
    #GET Video input for cam_feed
    videoReader = cv2.VideoCapture(video_input)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('test_output.avi', fourcc, 20.0, (640, 480))

    frame_count=0
    frames=[]
    while frame_count<clip_size:
        isCurrentFrameValid, currentImage = videoReader.read()
        frames.append(currentImage)
        frame_count=frame_count+1

        out.write(currentImage)
    videoReader.release()
    out.release()
    cv2.destroyAllWindows()

    device,c3d_network_n,ad_net=network_setup(ad_model_dir='/home/peter/Documents/actionID/AnomalyDetectionCVPR2018-Pytorch-master/short_60_low_mem/exps/model')


    #C3D
    dir_list,c3d_network,data_batch = cd3_extartion(os.getcwd()+'/'+'test_output.avi', device=device, features_dir="./test",clip_length=clip_size,
                                         train_frame_interval=1)
        #the data_batch is the input given to the C3D network. some how there is a added there so that the shape is [1, 3, 16, 244, 244] curently I get a shape of  [3, 16, 244, 244]
    print("Standered C3D methord complet")

    c3d_outputs,avg_segments=cd3_sigle_extartion(frames, c3d_network=c3d_network_n)
    print("Single C3D methord complet")

    #AD
    ad_model_dir = '/home/peter/Documents/actionID/AnomalyDetectionCVPR2018-Pytorch-master/short_60_low_mem/exps/model'
    features_dir="./test"
    video_parth=os.getcwd()+'/'+'test_output.avi'
    batch_y_pred = AD_perdiction(ad_model_dir, dir_list,features_dir,video_parth, device=device)
    print("batch_y_pred ="+str(batch_y_pred))

    sigle_y_pred = AD_sigle_perdiction(ad_model_dir, avg_segments, device=device,lengths=16,network=ad_net)
    print("single_y_pred =" + str(sigle_y_pred))

    # if sigle_y_pred.any() !=batch_y_pred.any():
    #     print("Error perditions from batch and sigle run are not the same?")
    #     print("b_y_pred =" + str(batch_y_pred))
    #     print("s_y_pred =" + str(sigle_y_pred))







if __name__ == '__main__':
    testing()