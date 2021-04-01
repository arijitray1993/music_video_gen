# Music Video Generation using GAN's

<a href="http://www.youtube.com/watch?feature=player_embedded&v=9YrvgymhIJk
" target="_blank"><img src="http://img.youtube.com/vi/9YrvgymhIJk/0.jpg" 
alt="generated video sample" width="400" border="10" /></a>


## Requirements

```bash
pip install librosa==0.7.2 torch==0.4.1 torchvision pillow scipy moviepy tqdm 
```


## Train on custom data 
To train on your own music videos, you will need to create a folder of music videos and their corresponding mp3's. 
To do so, place music videos in a folder and do:

`ffmpeg -i name_of_video.mp4 name_of_video.mp3` for each video mp4 file. 
I will upload a script soon that automates this, but this has to be manual for now, sorry.

Then, simply run:
```bash
cd Progressive-GAN-pytorch

python train.py --path /path/to/folder/containing/video&music
```


## Test on your music using pretrained checkpoint

To test on your own music file using a pretrained model, simply run:

```bash
cd Progressive-GAN-pytorch
python music2video.py --input_file /path/to/music.mp3 --outname /desired/output/videoname.mp4 --checkpoint optional/path/to/your_checkpoint_generator.pt
# if you wish to use our checkpoint, do not use the --checkpoint argument
```

### Citation

If you use this code, please consider citing:

```latex
@misc{raymusicvideogen,
      title={Swing Dance Video Generation using ProgressiveGAN}, 
      author={Arijt Ray},
      year={2020},
      url={https://github.com/arijitray1993/music_video_gen/}
}
```


### Acknowledgements
_The NVIDIA Progressive GAN code was modified from https://github.com/odegeasslbc/Progressive-GAN-pytorch_







