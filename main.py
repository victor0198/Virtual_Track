from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep, time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as image_keras_preprocessing
from tensorflow.keras.models import load_model
from SerialHandler import SerialHandler
from SerialHandler import ReadThread
from SerialHandler import FileHandler
import threading
import serial
from PIL import Image
from PIL.ImageDraw import ImageDraw


camera = cv2.VideoCapture(0)
# init models
print("loading keras angle")
model_steering = load_model('models/steering__5.h5')
print("loading keras speed")
model_speed = load_model('models/road_3_final.h5')

# init camera
#camera = PiCamera()
#camera.resolution = (640, 480)
#camera.framerate = 32
#rawCapture = PiRGBArray(camera, size=(640, 480))
#sleep(0.1)

# init vars
angles_lst = []
road_value = 3
road_value_prev = 3
angle_float = 0

# init microcontroller connection
handler = SerialHandler()
thread = ReadThread(threading.Thread, serial.Serial('/dev/ttyACM0', 256000, timeout=1),
                    FileHandler('history.txt'),
                    True)
thread.start()
handler.sendEncoderPublisher(True)
handler.sendPidActivation(True)

# read directions
rd = open("Directions.txt", "r")
directions_list = rd.readlines()
directions_list = [x.strip() for x in directions_list]
print('DIRECTIONS:')
print(directions_list)
rd.close()

road_prediction = []
color = (0, 200, 0)

while True:
    return_value, frame = camera.read()
#    cv2.imshow('hz', frame)
    image = Image.fromarray(frame)
    image.save("frames/" + str(time()) + "_" + str(angle_float) + ".png")
#    print("image")
    # image preparations for keras prediction
#    image_prediction = image.crop((0, 300, 1640, 1032))
    image_prediction = image
    image_prediction = image_prediction.resize((110, 140), Image.ANTIALIAS)
    draw = ImageDraw(image_prediction)
    draw.rectangle((0, 131, 110, 140), fill=color)
#    image_prediction.save("frames/" + str(time()) + "_" + str(angle_float) + ".png")

    image_prediction_angles = image_keras_preprocessing.img_to_array(image_prediction)
    image_prediction_angles = np.expand_dims(image_prediction_angles, axis=0)

    image_speed = Image.fromarray(frame)
#    image_speed = image_speed.crop((0, 320, 1640, 1120))
    image_speed = image_speed.resize((130, 205), Image.ANTIALIAS)
    #image_speed.save("frames/" + str(time()) + "_" + str(angle_float) + ".png")
    image_prediction_speed = image_keras_preprocessing.img_to_array(image_speed)
    image_prediction_speed = np.expand_dims(image_prediction_speed, axis=0)

    # steering model
    rslt = model_steering.predict(image_prediction_angles)
    #predicted_angles = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1,
    #                     0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2,
    #                     0.3, 0.3, -0.3, 0, -0.3, 0.3]

    predicted_angles = [20, 15, 10, 0, -10, -15, -20, 20, 15, 10,
                        0, -10, -15, -20, 20, 15, 10, 0, -10, -15, -20, 20, 15, 10, 0, -10, -15,
                        -20]
    angle_float = 0
    for idx in range(28):
        angle_float += rslt[0][idx] * predicted_angles[idx]
    if len(angles_lst) > 2:
        angles_lst.pop(0)

    angles_lst.append(round(angle_float, 2))
    final_angle = round(sum(angles_lst) / len(angles_lst), 3)

    # speed model
    rslt2 = model_speed.predict(image_prediction_speed)
    speeds = [0.25, 0.30, 0.2, 0.35, 0.15, 0.35, 0.4]

    statuses = ["cross", "curve", "downhill", "intersection",
                "stop line", "straight road", "uphill"]

    # "cross", 0
    # "curve", 1
    # "downhill", 2
    # "intersection", 3
    # "stop line", 4
    # "straight road", 5
    # "uphill" 6

    gl_speed = 0
    for idx in range(7):
        gl_speed += rslt2[0][idx] * speeds[idx]
        if rslt2[0][idx] > 0.9:
            status = statuses[idx]

    road_prediction.append(rslt2[0].argmax())
    print("-----------road_prediction", road_prediction)
    if len(road_prediction) > 3:
        road_prediction.pop(0)

    road_value_prev = road_value
    rdv = np.array(road_prediction)
    road_value = np.bincount(rdv).argmax()
    print("-----------road_value", road_value)
    if road_value == 5:
        color = (0, 200, 0)

    if road_value == 4:
        if len(directions_list) > 0:
            new_direction = directions_list.pop(0)
            if new_direction == 'forward':
                color = (0, 200, 0)
            if new_direction == 'left':
                color = (200, 0, 0)
            if new_direction == 'right':
                color = (0, 0, 200)
            print("---------new_direction:", new_direction)

    print("---------color ", color)
    handler.sendMove(gl_speed, angle_float)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        handler.sendBrake(0.0)
#        handler.stop()
        break
    
