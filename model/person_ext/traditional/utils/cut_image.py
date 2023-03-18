import os
from PIL import Image
import numpy as np


def cut_image(path, cut_path, size):
	"""
	剪切图片
	:param path: 输入图片路径
	:param cut_path: 剪切图片后的输出路径
	:param size: 要剪切的图片大小
	:return:
	"""
	for (root, dirs, files) in os.walk(path):
		temp = root.replace(path, cut_path)
		if not os.path.exists(temp):
			os.makedirs(temp)
		for file in files:
			image, flag = cut(Image.open(os.path.join(root, file)))
			if not flag:
				Image.fromarray(image).convert('L').resize((size, size)).save(os.path.join(temp, file))

	pass


def cut(image):
	"""
	通过找到人的最小最大高度与宽度把人的轮廓分割出来，、
	因为原始轮廓图为二值图，因此头顶为将二值图像列相加后，形成一列后第一个像素值不为0的索引。
	同理脚底为形成一列后最后一个像素值不为0的索引。
	人的宽度也同理。
	:param image: 需要裁剪的图片 N*M的矩阵
	:return: temp:裁剪后的图片 size*size的矩阵。flag：是否是符合要求的图片
	"""

	image = np.array(image)

	# 找到人的最小最大高度与宽度
	# print(image)
	# print(image.sum(axis=1))
	# print(image.sum(axis=1)!=0)
	height_min = (image.sum(axis=1) != 0).argmax()  # 将一行叠加，找到第一个白色出现的行
	height_max = ((image.sum(axis=1) != 0).cumsum()).argmax()  # 将一行叠加，找到最后一个白色出现的行
	width_min = (image.sum(axis=0) != 0).argmax()
	width_max = ((image.sum(axis=0) != 0).cumsum()).argmax()
	head_top = image[height_min, :].argmax()
	# 设置切割后图片的大小，为size*size，因为人的高一般都会大于宽
	size = height_max - height_min
	temp = np.zeros((size, size))

	# 以头为中心，将width_max-width_min（宽）乘height_max-height_min（高，szie）的人的轮廓图，放在size*size的图片中央
	l1 = head_top - width_min
	r1 = width_max - head_top
	# 若宽大于高，或头的左侧或右侧身子比要生成图片的一般要大。则此图片为不符合要求的图片
	flag = False
	if size <= width_max - width_min or size // 2 < r1 or size // 2 < l1:
		flag = True
		return temp, flag
	# centroid = np.array([(width_max+width_min)/2,(height_max+height_min)/2], dtype='int')
	temp[:, (size // 2 - l1):(size // 2 + r1)] = image[height_min:height_max, width_min:width_max]

	return temp, flag


def gei(cut_path, data_path, size):
	"""
	生成步态能量图
	:param cut_path: 剪切后的图片路径
	:param data_path: 生成图片的路径
	:param size: 生成能量图大小
	:return:
	"""
	for (root, dirs, files) in os.walk(cut_path):
		temp = root.replace(cut_path, data_path)
		if not os.path.exists(temp):
			os.makedirs(temp)
		GEI = np.zeros([size, size])
		if len(files) != 0:
			for file in files:
				GEI += Image.open(os.path.join(root, file)).convert('L')
			GEI /= len(files)
			Image.fromarray(GEI).convert('L').resize((size, size)).save(os.path.join(temp, 'gei.png'))
	pass


if __name__ == '__main__':
	INPUT_PATH = r"D:\Development\Python\Pycharm\AI\09GaitRecognitionSystem\GaitRecognitionSystem\data\upload\124\silhouette"
	OUTPUT_PATH = r"D:\Development\Python\Pycharm\AI\09GaitRecognitionSystem\GaitRecognitionSystem\data\predata"
	# OUTPUT_PATH2 = r"D:\Development\Python\Pycharm\AI\09GaitRecognitionSystem\GaitRecognitionSystem\data"
	cut_image(INPUT_PATH, OUTPUT_PATH, 128)
	# gei(OUTPUT_PATH, OUTPUT_PATH2, 128)
