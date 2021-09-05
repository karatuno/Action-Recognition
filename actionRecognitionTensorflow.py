#!/usr/bin/env python
"""
@file actionRecognitionTensorFLow.py .

@author Ayush Saini
@email saini712@gmail.com
@brief Action inference (TensorFlow) from video folder.
@version 0.0.1
@date 2021-07-25
@copyright Copyright (c) 2021
"""
import argparse
import os

import albumentations as A

import cv2

import numpy as np

import tensorflow as tf

# construct the argument parser
parser = argparse.ArgumentParser()
# model path
parser.add_argument('-m', '--model-path', dest='model_path',
                    help='path to folder containg saved_model.pb')
# path of folder containing videos for inference
parser.add_argument('-f', '--folder-path', dest='folder_path',
                    help='path of inference videos')
args = vars(parser.parse_args())

# define transforms according to model input
transform = A.Compose([
    A.Resize(224, 224, always_apply=True),
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
    input_frames = np.array(frames) / 255.0
    # Add a batch axis to the to the sample video.
    model_input = tf.constant(input_frames, dtype=tf.float32)[tf.newaxis, ...]

    logits = model(model_input)['default'][0]
    # softmax layer to get probablities
    probabilities = tf.nn.softmax(logits)

    result = [(
            class_names[idx].strip(), f'{probabilities[idx] * 100:5.2f}%')
             for idx in np.argsort(probabilities)[::-1][:2]]
    return result


# read the class names from labels.txt
with open('labels.txt', 'r') as f:
    class_names = f.readlines()
    f.close()

# We are using pretrained Inflated3DConvnet https://arxiv.org/abs/1705.07750
loaded = tf.saved_model.load(args['model_path'], tags=[])
i3d = loaded.signatures['default']

video_folder_path = args['folder_path']
# iterate through all video and print top 2 inference
for video in os.listdir(video_folder_path):
    print(video + ' '+str(predict(load_video(video_folder_path+video), i3d)))
