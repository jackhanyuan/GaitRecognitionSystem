#! /usr/bin/python3
# coding=utf-8

import os
import cv2
import torch
from .model import MattingNetwork
from .inference import convert_video


def person_ext_rvm(vid, input_path, person_folder, frame_size_threshold=800):
	print(f"\t Start silhouette extraction.")

	silhouette_path = os.path.sep.join([person_folder, 'silhouette', vid])
	image_path = os.path.sep.join([person_folder, 'image', vid])
	os.makedirs(silhouette_path, exist_ok=True)
	os.makedirs(image_path, exist_ok=True)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# load mobilenetv3 model
	model = MattingNetwork('mobilenetv3').eval().to(device)
	model_path = os.path.sep.join([os.path.dirname(__file__), "work", "checkpoint", "rvm_mobilenetv3.pth"])
	model.load_state_dict(torch.load(model_path))

	# calc video input_resize
	cap = cv2.VideoCapture(input_path)
	frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	cap.release()
	if max(frame_width, frame_height) <= frame_size_threshold:
		input_resize = None
	else:
		if frame_width > frame_height:
			ratio = frame_width / frame_size_threshold
		else:
			ratio = frame_height / frame_size_threshold
		input_resize = (int(frame_width // ratio), int(frame_height // ratio))
	print(f"\t {input_resize=}")

	# output png_sequence
	convert_video(
		model,  # 模型，可以加载到任何设备（cpu 或 cuda）
		input_source=input_path,  # 视频文件，或图片序列文件夹
		# num_workers=1,  # 只适用于图片序列输入，读取线程
		input_resize=input_resize,  # [可选项] 缩放视频大小
		output_type='png_sequence',  # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
		output_background='default',  # [可选项] 定义输出视频或图片序列的背景, 默认"default", 可选 "green", "white", "image"
		output_composition=image_path,  # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
		output_alpha=silhouette_path,  # [可选项] 输出透明度预测
		downsample_ratio=None,  # 下采样比，可根据具体视频调节，或 None 自动下采样至 512px
		seq_chunk=4,  # 设置多帧并行计算
		progress=True  # 显示进度条
	)


# if __name__ == '__main__':
# 	input_path = "data\\upload\\14\\video\\b2092bb2.avi"
# 	person_folder = "data\\upload\\14"
# 	vid = "b2092bb2"
# 	person_ext_rvm(vid, input_path, person_folder)
