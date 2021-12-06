

import helpers
import os



import cv2
import random
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow import keras
import tensorflow as tf




import pickle
## TODO: Допишите импорт библиотек, которые собираетесь использовать


def get_standatr_signs():
    standart_signs_list = []

    return standart_signs_list


def load_final_model():
    model = tf.keras.models.load_model('output/simple_nn.model')
    #model = []
    return model


def one_hot_encode(label):
    one_hot_label_dictionary = {"no entry": [1, 0, 0, 0, 0, 0, 0, 0],
                                "pedestrian crossing": [0, 1, 0, 0, 0, 0, 0, 0],
                                "road works": [0, 0, 1, 0, 0, 0, 0, 0],
                                "movement prohibition": [0, 0, 0, 1, 0, 0, 0, 0],
                                "movement_prohibition": [0, 0, 0, 1, 0, 0, 0, 0],
                                "parking": [0, 0, 0, 0, 1, 0, 0, 0],
                                "stop": [0, 0, 0, 0, 0, 1, 0, 0],
                                "give way": [0, 0, 0, 0, 0, 0, 1, 0],
                                "artificial roughness": [0, 0, 0, 0, 0, 0, 0, 1]}
    one_hot_encoded = one_hot_label_dictionary[label]
    return one_hot_encoded


def standardize_input(image):

    image = cv2.resize(image, (32, 32))
    image = image.flatten()
    image = image.reshape((1, image.shape[0]))
    standard_im = image  # по умолчанию, функция не меняет изображени
    return standard_im


def predict_label(image, model, standart_signs):
    standard_im = standardize_input(image)

    preds = model.predict(standard_im)

    i = preds.argmax(axis=1)[0]
    lb = pickle.loads(open('output/simple_nn_lb.pickle', "rb").read())

    predicted_label = lb.classes_[i]
    #print(predicted_label)
    text = "{}: {:.2f}%".format(predicted_label, preds[0][i] * 100)
    print(text)

    encoded_label = one_hot_encode(predicted_label)

    return encoded_label