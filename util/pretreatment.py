#! /usr/bin/python3
# coding=utf-8

import os
import cv2
import time
import numpy as np
import argparse
import pickle
from PIL import Image
from util.general import del_file
from model.person_cls.classification import Classification

classfication = Classification()
parser = argparse.ArgumentParser(description='Pretreatment')
parser.add_argument('--img_size', default=64, type=int, help='Image resizing size. Default 64')
parser.add_argument('--augment', default=False, type=bool, help='Image Horizontal Flip Augmented Dataset')
parser.add_argument('--is_people', default=True, type=bool, help='judge is people')
opt = parser.parse_args()


def is_people(img):
    if opt.is_people:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        class_name, probability = classfication.detect_image(img)
        if class_name == 'people' and probability > 0.5:
            return True
        else:
            return False
    return True


def imgs_to_pickle(vid, person_folder, save_cut_img=False, pixel_threshold=800):
    # silhouette_dir = person_folder
    silhouette_dir = os.path.sep.join([person_folder, "silhouette", vid])

    print(f"\t Walk to {person_folder}.")
    print(f"\t {save_cut_img=} {pixel_threshold=}")
    out_dir = os.path.sep.join([person_folder, "pkl", vid, "default"])
    all_imgs_pkl = os.path.sep.join([out_dir, '{}.pkl'.format(vid)])
    if os.path.exists(all_imgs_pkl):
        return len(os.listdir(silhouette_dir))

    count_frame = 0
    all_imgs, flip_imgs = [], []
    frame_list = sorted(os.listdir(silhouette_dir))
    for frame_name in frame_list:
        if frame_name.lower().endswith(('png', 'jpg')):  # filter png files
            frame_path = os.path.join(silhouette_dir, frame_name)
            image_path = os.path.sep.join([person_folder, "image", vid, frame_name])

            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            img = cut_img(img, 128, frame_name, pixel_threshold)  # cut img_size为128，便于后面is_people()判断
            img_copy = img

            if img is None:
                print('\t RM:', frame_name)
                os.remove(frame_path)
                os.remove(image_path)
                continue
            if is_people(img):
                # resize
                ratio = img.shape[1] / img.shape[0]
                img = cv2.resize(img, (int(opt.img_size * ratio), opt.img_size), interpolation=cv2.INTER_CUBIC)

                # Save the img
                all_imgs.append(img)
                count_frame += 1

                if save_cut_img:
                    cut_img_folder = os.path.sep.join([person_folder, "cut_img", vid])
                    os.makedirs(cut_img_folder, exist_ok=True)
                    cut_img_path = os.path.sep.join([cut_img_folder, "{0:.8f}-".format(time.perf_counter()) + frame_name])
                    cv2.imwrite(cut_img_path, img_copy)

                if opt.augment:
                    flip_img = cv2.flip(img, 1)  # 水平翻转，扩充数据
                    flip_imgs.append(flip_img)
                    print('\t augment:', frame_path)

            else:
                print('\t no people:', frame_name)
                print('\t RM:', frame_name)
                os.remove(frame_path)
                os.remove(image_path)
                # cv2.imwrite('people/false/' + "{0:.8f-}".format(time.perf_counter()) + frame_name, img)

    all_imgs = np.asarray(all_imgs + flip_imgs)

    if count_frame > 0:
        os.makedirs(out_dir, exist_ok=True)
        pickle.dump(all_imgs, open(all_imgs_pkl, 'wb'))
        print(f"\t pkl save path: {all_imgs_pkl}")

    # Warn if the sequence contains less than 5 frames
    if count_frame < 5:
        print('\t Seq:{}, less than 5 valid data.'.format(vid))

    return count_frame


def cut_img(img, img_size, frame_name, pixel_threshold=0):
    # A silhouette contains too little white pixels
    # might be not valid for identification.
    if img is None or img.sum() <= 10000:
        print(f'\t {frame_name} has no data.')
        return None

    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y > pixel_threshold).argmax(axis=0)  # the line pixels more than pixel_threshold, it will be counted
    y_btm = (y > pixel_threshold).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]

    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    ratio = img.shape[1] / img.shape[0]
    img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)

    # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        print(f'\t{frame_name} has no center.')
        return None

    # Get the left and right points
    half_width = img_size // 2
    left = x_center - half_width
    right = x_center + half_width
    if left <= 0 or right >= img.shape[1]:
        left += half_width
        right += half_width
        _ = np.zeros((img.shape[0], half_width))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right].astype('uint8')
    return img


def datasets_to_pkl(input_path):
    print(f"Walk to {input_path}.")
    id_list = os.listdir(input_path)
    id_list.sort()
    for _id in id_list:
        seq_type = os.listdir(os.path.join(input_path, _id))
        seq_type.sort()
        for _seq_type in seq_type[::1]:
            view = os.listdir(os.path.join(input_path, _id, _seq_type))
            view.sort()
            for _view in view[::1]:
                path = os.path.join(input_path, _id, _seq_type, _view)
                imgs_to_pickle(_seq_type, path)


def test_cut_img(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cut_img(img, 128, image_path, pixel_threshold=800)  # cut img_size为128，便于后面is_people()判断
    cv2.imwrite('test.png', img)

