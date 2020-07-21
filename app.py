import cv2
import torch
from torchvision import transforms
from transformer_net import TransformerNet
from imutils.video import VideoStream
from imutils import resize
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from flask import Flask, redirect, flash, request, url_for, send_file, jsonify
import os
from style_cam import style_frame, read_state_dict
import json
# import some shit
from flask_dropzone import Dropzone
import re
import utils
import random

ALLOWED_EXTENSIONS = set(['bmp', 'png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.secret_key = "secret key"
path = os.path.abspath(os.curdir)


if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

# Load the model
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('<<< Using CUDA >>>')
else:
    device = torch.device('cpu')
    print('CUDA device not available.')
    print('<<< Using CPU >>>')

dropzone = Dropzone(app)
app.config.update(
    UPLOADED_PATH='uploads',
    DROPZONE_MAX_FILE_SIZE=10000,
    DROPZONE_INPUT_NAME='data',
    DROPZONE_MAX_FILES=1,
)

IMAGE_FOLDER = os.path.join('static', 'images')

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload():
    if request.method == 'POST':
        input_img = request.files.getlist("file") # request.files['img'] 
        # style = request.form['style']
        return send_file(input_img, mimetype='image/jpeg') and redirect(url_for('style'))
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def style():
    # some crappy checks
    print([i for i in request.form.keys()])
    response = {'Status': None,
                'Message':None,
                'file':None}
    print(request.url)
    if 'file' not in request.files:
        response['Status'] = 'Failed'
        response['Message'] = 'No file attached to the request.'
        return json.dumps(response)
    # request file and style
    file = request.files['file']
    content_image = utils.load_image(file)
    style = request.form['style']
    model_get = style
    # some crappy checks
    if file.filename == '':
        response['Status'] = 'Failed'
        response['Message'] = 'No file attached to the request.'
        return json.dumps(response)
    if file and allowed_file(file.filename):
        content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)
        model_get = str(model_get)
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(model_get)
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            output = style_model(content_image).cpu()
            a = random.randint(1, 101)
            img_path = str("static/images/output_{}.jpg".format(a))
        utils.save_image(img_path, output[0])
        image_k = str("output_{}.jpg".format(a))

        get_image = os.path.join(app.config['UPLOAD_FOLDER'], image_k)
        print("Done")

    else:
        response['Status'] = 'Failed'
        response['Message'] = 'Unacceptable file type.'
        return json.dumps(response)
    return render_template("index.html", get_image=get_image)


if __name__== '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))