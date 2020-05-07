import sys
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import random
import pdb
import moviepy.editor as mpy
from scipy.misc import toimage
import argparse

sys.path.append("../scripts")

from process_music import process_music, musicality_noise_vectors

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from progan_modules import Generator

class music_data(Dataset):
    def __init__(self, audio_feats, noise_vectors, frame_batch=1):
        self.audio_feats = audio_feats
        self.noise_vectors = noise_vectors

        '''
        if frame_batch>1:
            batched_audio_feats = []
            entry=[]
            for a_f, n_f in zip(self.audio_feats, self.noise_vectors):
                entry.append((a_f, n_f))
                if len(entry)==frame_batch:
                    entry = list(zip(*entry))
                    batched_audio_feats.append((np.average(entry[0], axis=0), np.average(entry[1], axis=0)))
                    entry=[]
            self.audio_feats = batched_audio_feats
        else:
            self.audio_feats = list(zip(audio_feats, noise_vectors))
        '''

        batched_audio_feats = []
        entry=[]
        for a_f, n_f in zip(self.audio_feats, self.noise_vectors):
            entry.append((a_f, n_f))
            #if len(entry)==frame_batch:
            entry2 = list(zip(*entry))
            gaussian = np.linspace(-3,3, len(entry2[0]))
            gaussian = np.exp(-gaussian*gaussian)
            gaussian = np.repeat(gaussian[:, np.newaxis], 128, axis=1)
            #pdb.set_trace()
            batched_audio_feats.append((np.sum(gaussian*entry2[0], axis=0), entry2[1][0]))
            if len(entry)== 5:
                entry = entry[1:]
            #entry=[len()]
        self.audio_feats = batched_audio_feats

        


    def __getitem__(self, idx):
        return self.audio_feats[idx]
    
    def __len__(self):
        return len(self.audio_feats)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Generate video for a music file')

    parser.add_argument('--input_file', type=str, default="../data/test_music/DMajor_backing_improv_jam_040420.mp3", help='path of the input music file')
    parser.add_argument('--outname', type=str, default="../data/output_video/test_video.mp4")
    parser.add_argument('--checkpoint', type=str, default='trial_test1_2020-04-23_1_59/checkpoint/300000_g.model')
    args = parser.parse_args()

    ########## parameters #######################################################
    batch_size = 1
    fps = 10
    frame_batch=1 # number of frames the model is trained to generate at once
    checkpoint_file = args.checkpoint
    input_file = args.input_file
    outname = args.outname
    '''
    checkpoint_file = "trial_test1_2020-04-23_1_59/checkpoint/300000_g.model"
    
    input_file = "../data/test_music/DMajor_backing_improv_jam_040420.mp3" # "../data/test_music/OSR_uk_000_0020_8k.wav"
    outname="../data/output_video/test_video.mp4"
    '''
    device = torch.device("cuda:0")
    ###############s##############################################################
    
    # load song mp3 file
    spec, specm, gradm, audio_feats = process_music(audio_file=input_file, fps = fps, all_feats=True)
    print(audio_feats.shape)
    
    # make smooth music-guided noise vectors for feeding into Gen along with audio feats
    noise_vectors = musicality_noise_vectors(specm, gradm)

    # make audio dataloader
    data = music_data(audio_feats, noise_vectors, frame_batch=frame_batch)
    data_loader = DataLoader(data, shuffle=False, batch_size=batch_size,
                                 num_workers=4)

    
    # load generator model
    g_running = Generator(in_channel=128, out_channel=frame_batch*3, input_code_dim=256, pixel_norm=False, tanh=False).to(device)
    g_running.load_state_dict(torch.load(checkpoint_file))    
    g_running.train(False)
    
    #pass audio feats through generator
    print("Generating Video...")
    all_frames=[]
    alpha = 1
    step = 6 #step 6 does full resolution of 256x256 images
    #detransform = transforms.Denormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    noise_const = torch.randn(batch_size, 128).to(device)
    for feats, noise in tqdm(data_loader):
        #print(label.size())
        #label = next(data_loader) # label is the audio feat
        #pdb.set_trace()
        gen_z = torch.cat((feats.to(device).float(), noise.to(device).float()), dim=1)#concatenate with a noise vector for creativity.  noise.float().to(device)
        images = g_running(gen_z, step=step, alpha=alpha).data.cpu()

        images = images.view(images.shape[0]*frame_batch, 3, images.shape[-2], images.shape[-1]).numpy()

        all_frames.extend([np.array(toimage(image)) for image in images]) #images are potentially not denormalized, need to fix this. 


    # save frames into video file
    aud = mpy.AudioFileClip(input_file, fps = 44100)
    clip = mpy.ImageSequenceClip(all_frames, fps=fps)
    clip = clip.set_audio(aud)
    clip.write_videofile(outname, audio_codec='aac')




