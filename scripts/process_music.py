from collections import defaultdict
import librosa
import cv2
import numpy as np
import os
import pickle as pkl
import pdb
import h5py
from tqdm import tqdm
import random

# sr is sample rate : raw samples per second
# fps is frames per second
# len(spectrogram output) should match total number of frames
# len of spec output depends on hop length 
# when computing spectogram, we need to adjust hop length to match fps of video. then total frames read by cv2 and total sound points by librosa will match up. 

AUDIO_EXTENSIONS = ['mp3','wav', 'aac']
VIDEO_EXTENSIONS = ['mov', 'mp4']

## stuff for musicality vectors ###
tempo_sensitivity = 0.25
jitter=0.5
truncation=1
###################################

def process_music(audio_file, fps, all_feats=False):
    # takss in a music file, makes it into spectral features for a given fps 
    print("Processing Audio ...")
    #read song, sr is sample rate
    y, sr = librosa.load(audio_file)

    #create spectrogram
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=127,fmax=8000, hop_length=round(sr/fps))

    #get mean power at each time point
    specm=np.mean(spec,axis=0)

    #compute power gradient across time points
    gradm=np.gradient(specm)

    #set max to 1
    gradm=gradm/np.max(gradm)

    #set negative gradient time points to zero 
    gradm = gradm.clip(min=0)

    audio_feats = np.concatenate((spec, gradm[np.newaxis, :]), axis=0).transpose() # should be time_points x dim_feats

    if all_feats:
        return spec, specm, gradm, audio_feats
    else:
        return audio_feats

def new_jitters(jitter):
    #Function adapted from https://github.com/msieg/deep-music-visualizer
    jitters=np.zeros(128)
    for j in range(128):
        if random.uniform(0,1)<0.5:
            jitters[j]=1
        else:
            jitters[j]=1-jitter        
    return jitters

def new_update_dir(nv2,update_dir):
    #Function adapted from https://github.com/msieg/deep-music-visualizer
    
    for ni,n in enumerate(nv2):                  
        if n >= 2*truncation - tempo_sensitivity:
            update_dir[ni] = -1  
                        
        elif n < -2*truncation + tempo_sensitivity:
            update_dir[ni] = 1   
    return update_dir

def musicality_noise_vectors(specm, gradm, dim=128):
    # run audio file through process_music() with all_feats=True to get specm and gradm. Function adapted from https://github.com/msieg/deep-music-visualizer
    
    #initialize first noise vector
    nv1 = np.random.rand(dim)
    noise_vectors=[nv1]
    #initialize previous vectors (will be used to track the previous frame)
    nvlast=nv1
    #initialize the direction of noise vector unit updates
    update_dir=np.zeros(128)
    for ni,n in enumerate(nv1):
        if n<0:
            update_dir[ni] = 1
        else:
            update_dir[ni] = -1

    #initialize noise unit update
    update_last=np.zeros(128)
    
    for i in tqdm(range(len(gradm))):      
        #update jitter vector every 100 frames by setting ~half of noise vector units to lower sensitivity
        if i%200==0:
            jitters=new_jitters(jitter)
        #get last noise vector
        nv1=nvlast
        #set noise vector update based on direction, sensitivity, jitter, and combination of overall power and gradient of power
        update = np.array([tempo_sensitivity for k in range(128)]) * (gradm[i]+specm[i]) * update_dir * jitters 
        #smooth the update with the previous update (to avoid overly sharp frame transitions)
        update=(update+update_last*3)/4
        #set last update
        update_last=update
        #update noise vector
        nv2=nv1+update
        #append to noise vectors
        noise_vectors.append(nv2)
        #set last noise vector
        nvlast=nv2
        #update the direction of noise units
        update_dir=new_update_dir(nv2,update_dir)

    return noise_vectors


def process_video(video_file, fps=None, frame_size = (256, 256)):
    print("Processing Video ...")
    # takes in a video, makes it into frames with a given fps, and frame size
    video = cv2.VideoCapture(video_file)
    
    v_fps = video.get(cv2.CAP_PROP_FPS)
    fps = v_fps
    print("FPS of video: "+str(v_fps))

    all_frames = []
    i_cnt = 0
    if fps!=None:
        skip_length = max(round(v_fps/fps), 1)
    while(True):
        i_cnt+=1
        ret, frame = video.read()

        if ret== False:
            break;

        if fps!=None:
            if i_cnt%skip_length!=0:
                continue;

        frame = cv2.resize(frame, (frame_size))
        all_frames.append(frame)
        #pdb.set_trace()
        print(i_cnt, end='\r', flush=True)

    return np.asarray(all_frames), fps


def make_train_dataset(video_files, audio_files, base_folder=""):
    # takes in a list of video files with their associalted audio files
    # and makes a dict of music featues and video frames for each song

    audio_video_dict = dict()

    for v_file, a_file in zip(video_files, audio_files):
        
        video_frames, fps = process_video(os.path.join(base_folder, v_file), fps = None) # should be n_frames x frames_size x frame_size x 3 (rgb)

        spec, specm, gradm, audio_features = process_music(os.path.join(base_folder, a_file), fps = fps, all_feats=True)  #should be n_frames x feat_dim

        noise_vectors = musicality_noise_vectors(specm, gradm)

        print(len(video_frames), len(audio_features), len(noise_vectors))

        frame_count = min(len(video_frames), len(audio_features))

        video_frames = video_frames[:frame_count]
        audio_features = audio_features[:frame_count]
        noise_vectors = audio_features[:frame_count]

        audio_video_dict[v_file]=(audio_features, video_frames, noise_vectors)


    return audio_video_dict

def get_audio_video_frame_data(folder):
    #folder = "/dataSRI/ARIJIT/music_video_gan/data/music_video_data/trance_animations"

    audio_video_files = os.listdir(folder)

    audio_video_pairing = defaultdict(dict)
    for f in audio_video_files:

        f_name = f.split(".")[0]
        f_ext = f.split(".")[-1]

        if f_ext in AUDIO_EXTENSIONS:
            audio_video_pairing[f_name]["audio"] = f

        if f_ext in VIDEO_EXTENSIONS:
            audio_video_pairing[f_name]["video"] = f

    all_video_files = []
    all_audio_files = []
    for f in audio_video_pairing:

        all_video_files.append(audio_video_pairing[f]["video"])
        all_audio_files.append(audio_video_pairing[f]["audio"])
        
    audio_video_feats = make_train_dataset(all_video_files, all_audio_files, base_folder=folder)

    return audio_video_feats


if __name__=="__main__":

    # folder containing videos and audios as input for now. 

    folder = "/dataSRI/ARIJIT/music_video_gan/data/music_video_data/trance_animations"
    experiment_name = "exp1"

    audio_video_files = os.listdir(folder)

    audio_video_pairing = defaultdict(dict)
    for f in audio_video_files:

        f_name = f.split(".")[0]
        f_ext = f.split(".")[-1]

        if f_ext in AUDIO_EXTENSIONS:
            audio_video_pairing[f_name]["audio"] = f

        if f_ext in VIDEO_EXTENSIONS:
            audio_video_pairing[f_name]["video"] = f

    all_video_files = []
    all_audio_files = []
    for f in audio_video_pairing:

        all_video_files.append(audio_video_pairing[f]["video"])
        all_audio_files.append(audio_video_pairing[f]["audio"])
        
    audio_video_feats = make_train_dataset(all_video_files, all_audio_files, base_folder=folder)

    '''# figure out saving huge files later
    hf = h5py.File('data/processed_data/music_video_training.h5')
    hf.create_dataset

    with open("", "wb") as f:
        np.save(f, audio_video_feats)

    '''
    

