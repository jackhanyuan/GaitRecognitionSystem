import os

conf = {
    "WORK_PATH": os.path.dirname(__file__),
    "CUDA_VISIBLE_DEVICES": "0",  # "0,1" "cpu"
    "ALLOWED_EXTENSIONS": {'mp4', 'avi', 'wmv', 'flv', 'mov'},
    "UPLOAD_FOLDER": os.path.sep.join([os.path.dirname(__file__), 'data', 'upload']),
    "TMP_FOLDER": os.path.sep.join([os.path.dirname(__file__), 'data', 'upload', 'tmp']),
    "STATIC_TMP_FOLDER": os.path.sep.join([os.path.dirname(__file__), 'static', 'tmp']),
    "DATASETS_FOLDER": os.path.sep.join([os.path.dirname(__file__), 'model', 'gait', 'datasets', 'real-pkl-64']),
    "PRE_METHOD": "rvm",  # "traditional"
    "ext_frame_size_threshold": 800,
    "cut_img_pixel_threshold": 800,
}