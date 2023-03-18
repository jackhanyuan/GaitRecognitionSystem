#! /usr/bin/python3
# coding=utf-8
import os
from database import create_folder
from .background_subtractor import background_subtractor


def vid_to_silhouette(video_path, person_folder):
    vid_to_silhouette_folder = os.path.sep.join([person_folder, 'silhouette'])
    create_folder(vid_to_silhouette_folder)
    print("-" * 80)
    print("Start video to silhouette")
    print("Video to silhouette save folder:", vid_to_silhouette_folder)

    background_subtractor(video_path, vid_to_silhouette_folder)

# print(str(count) + ' silhouettes save successful.')


# if __name__ == '__main__':
# 	video_path = '..\\data\\upload\\054-bg-01-180.avi'
# 	person_folder = '..\\data\\upload\\hanyuan'
# 	vid_to_silhouette(video_path, person_folder)
