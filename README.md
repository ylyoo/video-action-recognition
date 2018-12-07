# **Video Action Recognition Using Pytorch.**

This repo contains Pytorch implementations of the video action recognition task on 
the UCF101 Dataset. 

Currently, working on a ResNet + RNN architecture.


## Dataset ##

The UCF101 Dataset is of  contains 

![](/figures/UCF101_examples.PNG)

UCF101 has total 13,320 videos from 101 actions. Videos have various time lengths (frames) and different 2d image size; the shortest is 28 frames.

To avoid painful video preprocessing like frame extraction and conversion such as OpenCV or FFmpeg, here I used a preprocessed dataset from feichtenhofer directly. If you want to convert or extract video frames from scratch, here are some nice tutorials:

## Markdown Cheatsheet ##

https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
