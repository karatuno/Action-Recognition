#!/usr/bin/env python
"""
@file actionRecognitionPytorch.py .

@author Ayush Saini
@email saini712@gmail.com
@brief Action inference (Pytorch) from video folder.
@version 0.0.1
@date 2021-07-25
@copyright Copyright (c) 2021
"""
import argparse
import os

import albumentations as A

import cv2

import numpy as np

import torch

# construct the argument parser
parser = argparse.ArgumentParser()
# model path
parser.add_argument('-m', '--model-path', dest='model_path',
                    help='path to .pt model')
# path of folder containing videos for inference
parser.add_argument('-f', '--folder-path', dest='folder_path',
                    help='path of inference videos')
args = vars(parser.parse_args())

# define transforms according to model input
transform = A.Compose([
    A.Resize(129, 171, always_apply=True),
    A.CenterCrop(112, 112, always_apply=True),
    A.Normalize(mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989],
                always_apply=True)
])


def load_video(path):
    """Extract all frames from video and arrange in an array.

    Args:
        path ([String]): [path to video]

    """
    all_frames = []
    cap = cv2.VideoCapture(path)
    if (cap.isOpened() is False):
        print('Error while trying to read video. Please check path again')
    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret is True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(image=frame)['image']
            # append to array
            all_frames.append(frame)
        else:
            break
    cap.release()
    return all_frames


def predict(frames, model):
    """Predict top 2 actions taking in array of frames.

    Args:
        frames ([array]): [array of frames]

    """
    with torch.no_grad():  # we do not want to backprop any gradients
        input_frames = np.array(frames)
        # add an extra dimension
        input_frames = np.expand_dims(input_frames, axis=0)
        # transpose to get [1, 3, num_clips, height, width]
        input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
        # convert the frames to tensor
        input_frames = torch.tensor(input_frames, dtype=torch.float32)
        input_frames = input_frames.to(device)
        # forward pass to get the predictions
        outputs = model(input_frames)
        # softmax layer to get probablities
        percentage = torch.nn.functional.softmax(outputs, dim=1)[0]
        # get the prediction index
        _, preds = torch.sort(outputs.data, descending=True)
        result = [(
            class_names[idx].strip(), f'{percentage[idx] * 100:5.2f}%')
             for idx in preds[0][:2]]
        return result


# read the class names from labels.txt
with open('labels.txt', 'r') as f:
    class_names = f.readlines()
    f.close()
# get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# We are using pretrained Resnet3D18 https://arxiv.org/abs/1711.11248
r3d = torch.load(args['model_path'])
r3d = r3d.eval().to(device)
video_folder_path = args['folder_path']
# iterate through all video and print top 2 inference
for video in os.listdir(video_folder_path):
    print(video + ' '+str(predict(load_video(video_folder_path+video), r3d)))
