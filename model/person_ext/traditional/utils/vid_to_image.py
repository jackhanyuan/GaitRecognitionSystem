#! /usr/bin/python3
# coding=utf-8

import cv2
import os
from database import create_folder


def vid_to_image(video_path, person_folder, save_interval=2):
    vid_to_image_folder = os.path.sep.join([person_folder, 'image'])
    create_folder(vid_to_image_folder)

    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    print("-" * 80)
    print("Start video to image")
    print("Video fps:", fps)
    print("Video to image save interval:", save_interval)
    print("Video to image save folder:", vid_to_image_folder)

    count = 1
    while video.isOpened():
        ret, image = video.read()

        if not ret:
            break

        # image = cv2.resize(image, (512, 384))

        if count % save_interval == 0:
            name = "{0:03}".format(count // save_interval) + '.jpg'
            # print(tools + ' save successful.')
            vid_to_image_path = os.path.sep.join([vid_to_image_folder, name])
            cv2.imwrite(vid_to_image_path, image)
            print('\t' + str(count // save_interval) + ' images save successful.')

        count += 1
        cv2.waitKey(1)

    video.release()
