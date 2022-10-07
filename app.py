from flask import Flask, flash, request, redirect, url_for, render_template, send_file,jsonify
from werkzeug.utils import secure_filename
import os
import uuid
from Faiss import search
from PIL import Image
import io
from base64 import encodebytes
import glob
import random
from ResnetModel import get_model

app = Flask(__name__)
pred_model = get_model()

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img


@app.route('/', methods=['POST'])
def upload_file():
    idx_rd = random.randint(0,999999)
    UPLOAD_FOLDER = 'image/'+str(idx_rd)
    os.mkdir(UPLOAD_FOLDER)
    if 'file' not in request.files:
        return 'No file'
    #file = request.files['file']
    app.logger.info(request.files)
    upload_files = request.files.getlist('file')
    app.logger.info(upload_files)
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if not upload_files:
        flash('No selected file')
        return redirect(request.url)
   
   
    for file in upload_files:
        original_filename = file.filename
        extension = original_filename.rsplit('.', 1)[1].lower()
        filename = str(uuid.uuid1()) + '.' + extension
        file.save(UPLOAD_FOLDER+"/"+filename)
    result = search(UPLOAD_FOLDER,pred_model)
    print(result)
    encoded_template = get_response_image(result['template'])
    encoded_predict = get_response_image(result['predict'])
    files = glob.glob(UPLOAD_FOLDER+'/*')


    # xóa file sau khi xử lý
    # if(len(files)> 0):
    #     for f in files:
    #         os.remove(f)
    # os.rmdir(UPLOAD_FOLDER)
    return jsonify({'distance':str(result['dist']),'template':str(encoded_template).replace('\n',''),'predict':str(encoded_predict).replace('\n','')})