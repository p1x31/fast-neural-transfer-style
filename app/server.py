import cv2
import torch
from torchvision import transforms
from transformer_net import TransformerNet
from imutils.video import VideoStream
from imutils import resize
from fastai.vision.image import open_image, image2np
from PIL import Image as PILImage
import numpy as np
from flask import Flask, redirect, flash, request, url_for, send_file, jsonify,render_template
import os
from style_cam import style_frame, read_state_dict
import json
# import some shit
from flask_dropzone import Dropzone
import re
import utils
import random
import uvicorn

import onnx
import onnx_caffe2.backend

ALLOWED_EXTENSIONS = set(['bmp', 'png', 'jpg', 'jpeg', 'gif'])
app=Flask(__name__)
app.secret_key = "secret key"
path=os.path.abspath(os.curdir)


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

# weights = []
# model_list={}
# for idx,file in enumerate(os.listdir('models')):
#     wts=read_state_dict(os.path.join(path,'models',file))
#     weights.append(wts)
#     model_list.update({file:idx})
# model = TransformerNet()
# model.to(device)


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
    file=request.files['file']
    content_image = utils.load_image(file)
    style = request.form['style']
    model_get = style
    # some crappy checks
    if file.filename == '':
        response['Status'] = 'Failed'
        response['Message'] = 'No file attached to the request.'
        return json.dumps(response)
    if file and allowed_file(file.filename):
        # save orig file
        # TODO
        # filename = file.filename
        # img_path=os.path.join(path,'static/input',filename)
        #
        # file.save(img_path)
        # img = Image.open(BytesIO(file.read())).convert('RGB')
        # TODO 
        # content_image.save(img_path)
        #
        # print(img_path)
        # img=cv2.imread(img_path)
        # print some shit
        # print([k for k in request.form.keys()])
        # style = request.form['style']
        # print('\n\n',style,'\n\n')
        # not working loop
        # if style+'.pth' not in model_list.keys():
        #     idx_model = 0
        # else:
        #     idx_model = model_list[style+'.pth']
        # model.load_state_dict(weights[idx_model])
        # #output
        # output = style_frame(img,model,device)
        # cv2.imwrite('result.jpg',output[:,:,::-1])
        # return send_file('result.jpg')
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


# def encode(img):
#     img = (image2np(img.data) * 255).astype('uint8')
#     pil_img = PILImage.fromarray(img)
#     buff = BytesIO()
#     pil_img.save(buff, format="JPEG")
#     return base64.b64encode(buff.getvalue()).decode("utf-8")

# def _predict_single(fp):
#   img = open_image(fp)
#   pred = img.predict(_learn)
#   idx = pred.argmax()
#   label = _learn.data.classes[idx]
#   img_data = encode(img)
#   return { 'label': label, 'name': fp.filename, 'image': img_data }


# @app.route('/predict', methods=['POST'])
# def predict():
#   global _learn
#   files = request.files.getlist("files")
#   predictions = map(_predict_single, files)
#   return render_template('predict.html', predictions=predictions)

# Refactor this later please

# @app.route("/uploader", methods=['POST'])
# def upload_file():
#     # Check valid image and compress
#     try:
#     img = Image.open(BytesIO(request.files['imagefile'].read())).convert('RGB')
#         img = ImageOps.fit(img, (400, 400), Image.ANTIALIAS)
#     except:
#         error_msg = "Please choose an image file"
#         return render_template('index.html', **locals())
#     #TODO
#     #out_img = rundeeplearning(np.array(img), "Model/la_muse.ckpt")
#     img_io = BytesIO()
#     out_img.save(img_io, 'PNG')
#     # Display re-sized picture
#     png_output = base64.b64encode(img_io.getvalue())
#     processed_file = urllib.parse.quote(png_output)
#     return render_template('index.html', **locals())
# @app.route("/api/styles", methods=['GET'])
# def return_styles():
#     return jsonify(models_dict)

# @app.route("/api/uploader/<style>", methods=['POST'])
# def api_upload_file(style):
#     img = Image.open(BytesIO(request.files['imagefile'].read())).convert('RGB')
#     if min(img.size) > 400:
#         img = ImageOps.fit(img, (400, 400), Image.ANTIALIAS)
#     in_img = np.array(img)
#     model_file = models_dict[style]
#     out_img =  rundeeplearning(in_img, model_file)
#     return send_pil(out_img)

# Helpers utility functions
# def send_pil(pil_img):
#     img_io = BytesIO()
#     pil_img.save(img_io, 'JPEG', quality=90)
#     img_io.seek(0)
#     return send_file(img_io, mimetype='image/jpeg')

# def scale_img(style_path, style_scale):
#     scale = float(style_scale)
#     o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
#     scale = float(style_scale)
#     new_shape = (int(o0 * scale), int(o1 * scale), o2)
#     style_target = _get_img(style_path, img_size=new_shape)
#     return style_target

# def get_img(src, img_size=False):
#    img = src  # Already pill image
#    if not (len(img.shape) == 3 and img.shape[2] == 3):
#        img = np.dstack((img,img,img))
#    if img_size != False:
#        img = scipy.misc.imresize(img, img_size)
#    return img



# @app.route('/list',methods=['GET'])
# def list_of_models():
#     mlist = [os.path.splitext(m)[0] for m in model_list]
#     return jsonify({'list':mlist})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
