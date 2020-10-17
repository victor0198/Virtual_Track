from time import sleep, time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as image_keras_preprocessing
from tensorflow.keras.models import load_model
import threading
from PIL import Image
from PIL.ImageDraw import ImageDraw
import os


print("loading keras speed")
model_speed = load_model('models/road_3_final.h5')
road_prediction = []
road_value_prev = 0
road_value = 0
status = "straight road"

direct = "imgs"
for img in os.listdir(direct):
    print("img:" + img)
    im = Image.open("/home/dellen/Documents/bfmc/Virtual_Track/imgs/"+img)
    im.show()
    im = im.resize((130,205), Image.ANTIALIAS)
    image_prediction_speed = image_keras_preprocessing.img_to_array(im)
    image_prediction_speed = np.expand_dims(image_prediction_speed, axis=0)
    # speed model
    rslt2 = model_speed.predict(image_prediction_speed)
    speeds = [0.25, 0.30, 0.2, 0.35, 0.15, 0.35, 0.4]

    statuses = ["cross", "curve", "downhill", "intersection",
                "stop line", "straight road", "uphill"]
    gl_speed = 0
    for idx in range(7):
        gl_speed += rslt2[0][idx] * speeds[idx]
        if rslt2[0][idx] > 0.9:
            status = statuses[idx]
    print("gl_speed", gl_speed)
    print("status", status)
    road_prediction.append(rslt2[0].argmax())
    print("road_prediction", road_prediction)
    if len(road_prediction) > 3:
        road_prediction.pop(0)

    road_value_prev = road_value
    rdv = np.array(road_prediction)
    road_value = np.bincount(rdv).argmax()
    print("road_value", road_value)
    x = input()


    # if road_value == 5:
    #     color = (0, 200, 0)
    #
    # if road_value_prev == 5 and road_value == 4:
    #     if len(directions_list) > 0:
    #         new_direction = directions_list.pop(0)
    #         if new_direction == 'forward':
    #             color = (0, 200, 0)
    #         if new_direction == 'left':
    #             color = (200, 0, 0)
    #         if new_direction == 'right':
    #             color = (0, 0, 200)
    # print("color ", color)

