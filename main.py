#! /usr/bin/python3
# coding=utf-8

import os
import sys
from config import conf

print(f"{conf=}")
WORK_PATH = conf['WORK_PATH']
sys.path.append(WORK_PATH)
os.environ["CUDA_VISIBLE_DEVICES"] = conf["CUDA_VISIBLE_DEVICES"]
# print(f"{os.environ['CUDA_VISIBLE_DEVICES']=}")


from util.register import allowed_file, get_video_frame, run_opengait, register
from werkzeug.utils import secure_filename
from flask_toastr import Toastr
from flask import Flask, render_template, request, Response, redirect, url_for, flash

app = Flask(__name__)
toastr = Toastr(app)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

STATIC_TMP_FOLDER = conf["STATIC_TMP_FOLDER"]


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    try:
        if request.method == 'POST':
            person_name = secure_filename(request.form['name'])
            vid_file = request.files['regFile']
            if person_name and vid_file and allowed_file(vid_file.filename):
                tag, message = register(person_name, vid_file)
            else:
                tag = False
                message = "Invalid name or video, please check and re-upload."

            status = 'success' if tag else 'warning'
            flash(message, status)  # 显示message

        return redirect(url_for('index'))

    except Exception as error:
        print(error)


@app.route('/recognition', methods=['GET', 'POST'])
def gait_recognition():
    try:
        if request.method == 'POST':
            tag = False
            person_name = ""
            vid_file = request.files['recFile']
            if vid_file and allowed_file(vid_file.filename):
                tag, message = register(person_name, vid_file)
            else:
                message = "Invalid video, please check and re-upload."

            if tag:
                tmp_image_path = os.path.sep.join([STATIC_TMP_FOLDER, "image"])
                cut_image_folder = os.path.sep.join([STATIC_TMP_FOLDER, "cut_img"])
                images = [str("/static/tmp/image/" + image) for image in sorted(os.listdir(tmp_image_path))[5:-5:3]]
                cut_images = [str("/static/tmp/cut_img/" + image) for image in sorted(os.listdir(cut_image_folder))[5:-5:3]]
                return render_template('video.html', images=images, cut_images=cut_images)
            else:
                print(message)
                flash(message, 'warning')  # 显示message

        return redirect(url_for('index'))

    except Exception as error:
        print(error)


@app.route("/get_similarity")
def get_similarity():
    if request.headers.get('accept') == 'text/event-stream':

        return Response(run_opengait(), mimetype='text/event-stream')


@app.route("/video_feed")
def video_feed():
    return Response(get_video_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=5000)
