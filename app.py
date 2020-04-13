from werkzeug.utils import secure_filename
import joblib
import os
from flask import Flask, request,jsonify, render_template
import requests
import cv2
#from PIL import Image
#import tensorflow as tf
#import keras
import numpy as np

im=[]
app = Flask(__name__)

def process_img(imk):
    output1 = cv2.resize(imk, (150, 150))
    output1 = output1.astype('float')
    output1 /= 255.0
    print(type(output1))
    output1 = np.array(output1).reshape(-1, 150, 150, 1)
    #output1.shape
    dimData = np.prod(output1.shape[1:])
    output1 = output1.reshape(output1.shape[0], dimData)
    print(output1)
    classifer = joblib.load("First_Model.pk1")
    x=classifer.predict_classes(output1[[0], :])
    #return "hello"
    if x[0]==1:
        result="Patient have Pnemonia"
    else:
        result="patient doesn't have pnemonia"
    return jsonify({"data" : result })

    #output1.shape
    #output1[[0], :]

@app.route('/handle_form', methods=['POST'])
def handle_form():
    print("Posted file: {}".format(request.files['file']))
    file = request.files['file']
    #file.shape
    print(file)
    file.save(secure_filename("save.jpeg"))
    #im = cv2.imread(file)
    im=cv2.imread("save.jpeg",0)
    #print(im.shape)
    #print(im)
    result=process_img(im)
    #print(type(im))
    #plt.imshow(file)
    #plt.show()
    #print(file.shape)
    #file.save()
    #print(file)
    #cv2.imwrite(filename='saved_img.jpg')
    return result

@app.route("/")
def index():
   return "working"

if __name__ == "__main__":
    app.run()
