#! /usr/bin/python3
# coding=utf-8
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
import os


def find_the_max_box(boxes):
    # Find the max box.
    max_rec = 0
    x_max, y_max, w_max, h_max = 0, 0, 0, 0
    for box in boxes:

        (x, y, w, h) = box

        if w * h < 20000:
            continue

        if w > 100 and h > 200:
            if max_rec < w * h:
                max_rec = w * h
                (x_max, y_max, w_max, h_max) = box
    return x_max, y_max, w_max, h_max


def get_median_frame(video_path):
    # 读取视频
    video = cv2.VideoCapture(video_path)

    # 随机选择 25 frames
    frame_ids = video.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

    # 把上面选定好的frames 放在一个 array
    frames = []
    for fid in frame_ids:
        video.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = video.read()
        frames.append(frame)

    # 通过时间轴计算medianFrame
    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)

    resize_median_frame = cv2.resize(median_frame, (512, 384))
    # 灰度处理
    gray_median_frame = cv2.cvtColor(resize_median_frame, cv2.COLOR_BGR2GRAY)
    # 高斯滤波，(3, 3)表示高斯矩阵的长与宽都是3，标准差取0
    gray_median_frame = cv2.GaussianBlur(gray_median_frame, (3, 3), 0)

    # 显示median frame
    # cv2.imshow('median frame', gray_median_frame)
    # cv2.destroyAllWindows()

    video.release()

    return gray_median_frame


def find_people(frame, gray):
    # 获取所有检测框
    (cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_rec = 0
    # Find the max box.
    for c in cnts:
        if cv2.contourArea(c) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)

        if w > 25 and h > 50:
            if max_rec < w * h:
                max_rec = w * h
                (x_max, y_max, w_max, h_max) = cv2.boundingRect(c)
    # If exist max box.
    if max_rec > 0 and h_max > w_max:
        # cv2.rectangle(frame, (x_max, y_max), (x_max + w_max, y_max + h_max), (0, 255, 0), 2)
        if x_max > 20:  # To ignore some regions which contain parts of human body.
            nim = np.zeros([gray.shape[0] + 10, gray.shape[1] + 10], np.single)  # Enlarge the box for better result.
            nim[y_max + 5:(y_max + h_max + 5), x_max + 5:(x_max + w_max + 5)] = gray[y_max:(y_max + h_max),
                                                                                x_max:(x_max + w_max)]
            # Get coordinate positionp.
            ty, tx = (nim > 100).nonzero()
            sy, ey = ty.min(), ty.max() + 1
            sx, ex = tx.min(), tx.max() + 1
            h = ey - sy
            w = ex - sx

            # Normal human should be like this, the height shoud be greater than wideth.
            if h > w:
                cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)

    return frame


def find_people_by_hog(frame, gray):
    # initialize the HOG descriptor
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    exist_people = False

    while True:
        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # NMS非极大值抑制，去除重叠，保留概率最大的box
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        boxes = np.array([[x, y, xw - x, yh - y] for (x, y, xw, yh) in pick])

        # 寻找最大的box
        (x_max, y_max, w_max, h_max) = find_the_max_box(boxes)
        max_rec = w_max * h_max
        if max_rec > 20000:
            exist_people = True
            cv2.rectangle(frame, (x_max, y_max), (x_max + w_max, y_max + h_max), (0, 255, 0), 2)

        # cv2.imshow('Rectangle Frame', frame)
        # cv2.destroyAllWindows()

        return exist_people, frame


def background_subtractor(video_path, save_folder):
    """
	absdiff(median_frame, gray)
	:param video_path:
	:return:
	"""
    filename = os.path.basename(video_path)
    median_frame = get_median_frame(video_path)
    count = 0

    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        frame = cv2.resize(frame, (512, 384))

        # 灰度处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 高斯滤波，(3, 3)表示高斯矩阵的长与宽都是3，标准差取0
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply background subtraction method.
        # 当前帧与背景帧像素相减
        frame_delta = cv2.absdiff(median_frame, gray)

        # 使用阈值去噪
        # cv2.threshold(输入的图片, thresh阈值, maxval最大值, type阈值类型)
        # cv2.THRESH_BINARY 表示阈值的二值化操作，大于阈值使用maxval表示，小于阈值使用0表示
        thresh = cv2.threshold(frame_delta, 35, 255, cv2.THRESH_BINARY)[1]

        # 闭运算，先进行膨胀然后进行腐蚀操作,消除内部空洞
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        dilate = cv2.dilate(thresh, kernel_dilate, iterations=2)
        erode = cv2.erode(dilate, kernel_erode, iterations=2)

        # 检测人
        exist_people, frame = find_people_by_hog(frame, gray)

        if exist_people:
            count += 1
            save_name = filename.rsplit('-', 1)[0].lower() + '_' + "{0:03}".format(count) + '.jpg'
            save_path = os.path.sep.join([save_folder, save_name])
            cv2.imwrite(save_path, erode)
            print('\t' + str(count) + ' silhouettes save successful.')

    # Show results.
    # cv2.imshow("gray", gray)
    # cv2.imshow("detection", frame)
    # cv2.imshow("back", erode)
    # if cv2.waitKey(110) & 0xff == 27:
    # 	break

    video.release()
    cv2.destroyAllWindows()
    return count


def background_subtractor2(video_path):
    """
	absdiff(median_frame, gray)
	:param video_path:
	:return:
	"""
    median_frame = get_median_frame(video_path)
    # median_frame = None

    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        frame = cv2.resize(frame, (512, 384))

        # 灰度处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 高斯滤波，(3, 3)表示高斯矩阵的长与宽都是3，标准差取0
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if median_frame is None:
            median_frame = gray  # Set this frame as the background.
        # cv2.imwrite('median_frame.jpg', median_frame)

        # Apply background subtraction method.
        # 当前帧与背景帧像素相减
        frame_delta = cv2.absdiff(median_frame, gray)

        # 使用阈值去噪
        # cv2.threshold(输入的图片, thresh阈值, maxval最大值, type阈值类型)
        # cv2.THRESH_BINARY 表示阈值的二值化操作，大于阈值使用maxval表示，小于阈值使用0表示
        thresh = cv2.threshold(frame_delta, 35, 255, cv2.THRESH_BINARY)[1]

        # 闭运算，先进行膨胀然后进行腐蚀操作,消除内部空洞
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        dilate = cv2.dilate(thresh, kernel_dilate, iterations=2)
        erode = cv2.erode(dilate, kernel_erode, iterations=2)

        # 检测人
        exist_people, frame = find_people_by_hog(frame, gray)

        # Show results.
        cv2.imshow("gray", gray)
        cv2.imshow("detection", frame)
        cv2.imshow("back", erode)
        if cv2.waitKey(110) & 0xff == 27:
            break

    video.release()
    cv2.destroyAllWindows()


def background_subtractor3(video_path):
    """
	BackgroundSubtractorKNN
	BackgroundSubtractorMOG2
	:param video:
	:return:
	"""
    video = cv2.VideoCapture(video_path)
    history = 20  # 训练帧数

    model = cv2.createBackgroundSubtractorKNN(history=history)  # 背景减除器，设置阴影检测
    # model = cv2.createBackgroundSubtractorMOG2(history=history, detectShadows=True)

    frames = 0

    while True:
        res, frame = video.read()

        if not res:
            break

        frame = cv2.resize(frame, (512, 384))

        # 灰度处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 高斯滤波，(3, 3)表示高斯矩阵的长与宽都是3，标准差取0
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        fg_mask = model.apply(gray)  # 获取 foreground mask
        bk = model.getBackgroundImage(gray)

        if frames < history:
            frames += 1
            continue

        # 对原始帧进行膨胀去噪
        thresh = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)[1]
        # 闭运算，先进行膨胀然后进行腐蚀操作,消除内部空洞
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        dilate = cv2.dilate(thresh, kernel_dilate, iterations=2)
        erode = cv2.erode(dilate, kernel_erode, iterations=2)

        # 检测人
        exist_people, frame = find_people_by_hog(frame, gray)

        cv2.imshow("detection", frame)
        cv2.imshow("back", erode)
        cv2.imshow("bk", bk)
        if cv2.waitKey(110) & 0xff == 27:
            break

    video.release()
    cv2.destroyAllWindows()


def background_subtractor4(video_path):
    """
	1) Frame Differencing:
	This method is through the difference between two consecutive images to determine the presence of moving objects
	:param video_path:
	:return:
	"""

    # function for trackbar
    def nothing(x):
        pass

    # naming the trackbar
    cv2.namedWindow('diff')

    # creating trackbar for values to be subtracted
    cv2.createTrackbar('min_val', 'diff', 0, 255, nothing)
    cv2.createTrackbar('max_val', 'diff', 0, 255, nothing)

    # creating video element
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    # converting the image into grayscale image
    image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # getting the shape of the frame capture which will be used for creating an array of resultant image which will store the diff
    row, col = image1.shape
    res = np.zeros([row, col, 1], np.uint8)

    # converting data type integers 255 and 0 to uint8 type
    a = np.uint8([255])
    b = np.uint8([0])
    while True:
        ret, image2 = cap.read()

        if not ret:
            break

        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # getting threshold values from trackbar according to the lightning condition
        min_val = cv2.getTrackbarPos('min_val', 'diff')
        max_val = cv2.getTrackbarPos('max_val', 'diff')

        min_val = 0
        max_val = 255

        # using cv2.absdiff instead of image1 - image2 because 1 - 2 will give 255 which is not expected
        res = cv2.absdiff(image1, image2)

        cv2.imshow('image', res)

        # creating mask
        res = np.where((min_val < res) & (max_val > res), a, b)
        res = cv2.bitwise_and(image2, image2, mask=res)

        cv2.imshow('diff', res)

        # assigning new new to the previous frame
        image1 = image2

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# if __name__ == '__main__':
# video = '../data/upload/124-bkgrd-000.avi'
# video = '../data/upload/124-bg-01-000.avi'
# video = '../data/upload/083-bkgrd-036.avi'
# video = '../data/upload/083-bg-01-036.avi'
# video = '../data/upload/064-bkgrd-036.avi'
# video = '../data/upload/064-cl-01-036.avi'
# video = '../data/upload/054-bkgrd-180.avi'
# video = '../data/upload/054-bg-01-180.avi'
# background_subtractor3(video)
