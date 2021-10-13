import flask
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
from urllib.request import urlopen, Request
import ssl
import numpy as np
import matplotlib.cm as cm
import time
import cv2
from flask import render_template


app = flask.Flask(__name__)
model = load_model('Pneumonia.h5')
IMAGE_SIZE = (256, 256)
classifier_layer_names = [
    'flatten',
    'dense',
    'dense_1'
]


def load_image(image):
    flag = False
    if image.mode != "RGB":
        flag = True
    print(flag)
    img = image.resize(IMAGE_SIZE)
    img = img_to_array(img)
    img = img.reshape(1, 256, 256, 3)
    img = img.astype('float32')

    return img


preprocess_input = tf.keras.applications.xception.preprocess_input
decode_predictions = tf.keras.applications.xception.decode_predictions


def find_target_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name


def get_img_array(image, size):
    img = image.resize(IMAGE_SIZE, Image.ANTIALIAS)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = {}
    print(flask.request.form)
    params = flask.request.json
    if (params == None):
        params = flask.request.args
    if (params != None):
        context = ssl._create_unverified_context()
        req = Request(params.get("image"), headers={
                      'User-Agent': 'Mozilla/5.0'})
        img = Image.open(urlopen(req, context=context))
        img = img.convert('RGB')
        image = load_image(img)
        result = model.predict(image)

        processed_img = keras.preprocessing.image.img_to_array(img)

        img_array = preprocess_input(get_img_array(img, size=IMAGE_SIZE))

        heatmap = make_gradcam_heatmap(
            img_array, model, find_target_layer(
                model), classifier_layer_names
        )
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize(
            (processed_img.shape[1], processed_img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * 0.4 + processed_img
        superimposed_img = keras.preprocessing.image.array_to_img(
            superimposed_img)

        save_path = "pneumonia_classification_{}.jpg".format(
            str(round(time.time() * 1000)))
        superimposed_img.save("static/output/"+save_path)

        prediction = ""
        if result[0][0] == 1:
            prediction = "pneumonia"
        else:
            prediction = "normal"
        data["prediction"] = prediction
        data["output"] = save_path
    return flask.jsonify(data)


# start the flask app, allow remote connections
app.run(port=5000)
