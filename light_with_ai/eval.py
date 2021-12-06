import numpy as np
import cv2
import helpers
import os
import random
from tensorflow.keras.models import load_model
from tensorflow import keras
import tensorflow as tf
import pickle


def one_hot_encode(label):

    """ Функция осуществляет перекодировку текстового "названия" сигнала
     в список элементов, соответствующий выходному сигналу

     Входные параметры: текстовая метка
     Выходные параметры: метка ввиде списка

     Пример:
        one_hot_encode("red") должно возвращать:        [1, 0, 0, 0, 0]
        one_hot_encode("yellow") должно возвращать:     [0, 1, 0, 0, 0]
        one_hot_encode("green") должно возвращать:      [0, 0, 1, 0, 0]
        one_hot_encode("yellow_red") должно возвращать: [0, 0, 0, 1, 0]
        one_hot_encode("off") должно возвращать:        [0, 0, 0, 0, 1]

     """
    one_hot_encoded = []

    if label == "red":
        one_hot_encoded = [1, 0, 0, 0, 0]
    elif label == "yellow":
        one_hot_encoded = [0, 1, 0, 0, 0]
    elif label == "green":
        one_hot_encoded = [0, 0, 1, 0, 0]
    elif label == "yellow_red":
        one_hot_encoded = [0, 0, 0, 1, 0]
    elif label == "off":
        one_hot_encoded = [0, 0, 0, 0, 1]

    return one_hot_encoded


def standardize_input(rgb_image):
    rgb_image = cv2.resize(rgb_image, (32, 32))
    rgb_image = rgb_image.flatten()
    rgb_image = rgb_image.reshape((1, rgb_image.shape[0]))
    standard_im = rgb_image  # по умолчанию, функция не меняет изображения
    return standard_im

model = tf.keras.models.load_model('output/simple_nn.model')
lb = pickle.loads(open('output/simple_nn_lb.pickle', "rb").read())
def predict_label(rgb_image):
    standard_im = standardize_input(rgb_image)


    preds = model.predict(standard_im)
    #print(preds)

    i = preds.argmax(axis=1)[0]
    #print(i)

    predicted_label = lb.classes_[i]

    #text = "{}: {:.2f}%".format(predicted_label, preds[0][i] * 100)
    #print(text)
    print(predicted_label)
    #predicted_label = 'yellow'
    encoded_label = one_hot_encode(predicted_label)
    #print(encoded_label)# по умолчанию, говорит что на всех изображения жёлтый сигнал

    ## TODO: ваша функция распознавания сигнала светофора должна быть здесь.
    return encoded_label
