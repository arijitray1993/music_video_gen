
import torch
import torchvision.transforms as transforms
import pickle as pkl
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import sys
import math
sys.path.append("../scripts")
from process_music import get_audio_video_frame_data
import pdb

class music_video_frames(data.Dataset):

  def __init__(self, folder=None, transform=None, frame_batch=1):

    #make a dataset of music embeddings and image frames 
    # load directly from process music function instead of loading precomputed
    #with open("../data/processed_data/music_video_training.npy", "rb") as f:
      
      #music_frame_data = np.load(f)
      # contains dict of keys as songs and the values as lists of aligned audio feats and video image frames. 

    # frame_batch is number of frames to predict at once for learning continuity.

    music_frame_data = get_audio_video_frame_data(folder)

    self.audio2frames = []
    
    for key in music_frame_data:
      audio_feats, video_frames, noise_vectors = music_frame_data[key]

      entry=[]
      for a_f, v_f, n_f in zip(audio_feats, video_frames, noise_vectors):
        entry.append((a_f, v_f, n_f))
        entry2 = list(zip(*entry))
        gaussian = np.linspace(-3,3, len(entry2[0]))
        gaussian = np.exp(-gaussian*gaussian)
        gaussian = np.repeat(gaussian[:, np.newaxis], 128, axis=1)
        weighted_audio_feat= np.sum(gaussian*entry2[0], axis=0)

        self.audio2frames.append((weighted_audio_feat, [entry2[1][int(len(entry)/2)]], entry2[2][0]))
        if len(entry) == 5:
          entry=entry[1:] 
    
    self.transform = transform
    

  

  def __getitem__(self, idx):
    
    audio_feats, video_frames, noise_vectors = self.audio2frames[idx]
    
    video_frames = [Image.fromarray(video_frame) for video_frame in video_frames]
    if self.transform!=None:
      video_frames = [self.transform(video_frame) for video_frame in video_frames]
    
    #audio_feat = np.average(audio_feats, axis=0)
    #noise_vector = np.average(noise_vectors, axis=0)

    new_video_frame = torch.stack(video_frames, dim=0)
    size = new_video_frame.size()
    new_video_frame = new_video_frame.view(-1,size[-2], size[-1])

    return new_video_frame, audio_feats, noise_vectors


  def __len__(self):

    return len(self.audio2frames)


