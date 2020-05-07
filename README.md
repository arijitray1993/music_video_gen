# Music Video Generation using Progressive GAN

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
python music2video.py --input_file /path/to/music.mp3 --outname /desired/output/videoname.mp4 
# if you wish to use our checkpoint
```
or
```bash
python music2video.py --input_file /path/to/music.mp3 --outname /desired/output/videoname.mp4 --checkpoint /path/to/your/trained/checkpoint
# if you wish to use your own trained checkpoint
```








