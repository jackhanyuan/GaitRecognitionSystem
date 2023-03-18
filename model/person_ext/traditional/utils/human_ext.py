#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 02:29:03 2021
@author: minhdoan
https://github.com/MinhBoll/GaitSet-Work/blob/main/human_extractor.py
"""

from __future__ import print_function
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import time
import os

# Load pretrained model
model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
# Segment people only for the purpose of human silhouette extraction
people_class = 15

# Evaluate model
model.eval()
print("Model has been loaded.")

blur = torch.FloatTensor([[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]]) / 16.0

# Use GPU if supported, for better performance
if torch.cuda.is_available():
    model.to('cuda')
    blur = blur.to('cuda')

# Apply preprocessing (normalization)
preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Function to create segmentation mask
def makeSegMask(img):
    # Scale input frame
    frame_data = torch.FloatTensor(img) / 255.0

    input_tensor = preprocess(frame_data.permute(2, 0, 1))

    # Create mini-batch to be used by the model
    input_batch = input_tensor.unsqueeze(0)

    # Use GPU if supported, for better performance
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]

    segmentation = output.argmax(0)

    bgOut = output[0:1][:][:]
    a = (1.0 - F.relu(torch.tanh(bgOut * 0.30 - 1.0))).pow(0.5) * 2.0

    people = segmentation.eq(torch.ones_like(segmentation).long().fill_(people_class)).float()

    people.unsqueeze_(0).unsqueeze_(0)

    for i in range(3):
        people = F.conv2d(people, blur, stride=1, padding=1)

    # Activation function to combine masks - F.hardtanh(a * b)
    combined_mask = F.relu(F.hardtanh(a * (people.squeeze().pow(1.5))))
    combined_mask = combined_mask.expand(1, 3, -1, -1)

    res = (combined_mask * 255.0).cpu().squeeze().byte().permute(1, 2, 0).numpy()

    return res


def frame_capture(file):
    # Loads video file into CV2
    video = cv2.VideoCapture(file)

    # Get video file's dimensions
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    # Creates output video file
    destination = 'asilla-data/preprocessed_videos_probe/'
    des_path = os.path.join(destination, os.path.basename(file))
    try:
        if not os.path.exists(destination):
            os.mkdir(destination)
    except OSError:
        print('Error: Creating directory of data')

    print(f"Writing to {des_path}")
    out = cv2.VideoWriter(des_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    prev_frame_time = 0
    new_frame_time = 0

    while video.isOpened:
        # Read each frame one by one
        success, img = video.read()

        # Run if there are still frames left
        if success:

            # Apply background subtraction to extract foreground (silhouette)
            mask = makeSegMask(img)

            # Apply thresholding to convert mask to binary map
            ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Write processed frame to output file
            out.write(thresh)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(fps)
            print(fps)
            # cv2.rectangle(mask, (10, 2), (100,20), (255,255,255), -1)
            # cv2.putText(mask, fps, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

            # Show extracted silhouette only, by multiplying mask and input frame
            final = cv2.bitwise_and(thresh, img)

            """ Show current frame
			cv2.imshow('Silhouette Mask', mask)
			cv2.imshow('Extracted Silhouette', final)
			
			# Allow early termination with Esc key
			key = cv2.waitKey(10)
			if key == 27:
				break"""
        # Break when there are no more frames
        else:
            break

    # Release resources
    # cv2.destroyAllWindows()
    video.release()
    out.release()


if __name__ == '__main__':
    data_path = "asilla-data/videos_probe"
    for file in os.listdir(data_path):
        if file.endswith(".mp4"):
            print(f"Processing {file}")
            path = os.path.join(data_path, file)
            frame_capture(path)

    print("DONE!!!")
