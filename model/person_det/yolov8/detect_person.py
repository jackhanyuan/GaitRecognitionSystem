#! /usr/bin/python3
# coding=utf-8

import cv2
# pip install ultralytics
from ultralytics import YOLO

model = YOLO("model/person_det/yolov8/yolov8s-seg.pt")
# model = YOLO("model/person_det/yolov8/yolov8s.pt")


def yolov8_detect_person(img, label):
    results = model.predict(source=img, imgsz=640, conf=0.80, iou=0.45, classes=[0])
    person_count = (results[0].boxes.cls == 0).sum()
    if person_count > 1:
        label = ''

    # labels : True -> YOLOv8 original label; '' or False -> no label;
    res_plotted = results[0].plot(labels=label)
    # cv2.imwrite("test.jpg", res_plotted)

    return res_plotted


if __name__ == '__main__':
    img = cv2.imread('./test.jpg', cv2.IMREAD_COLOR)
    yolov8_detect_person(img, label='')