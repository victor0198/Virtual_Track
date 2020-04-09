#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
self_driving = True

import sys
import argparse
import pyglet
from PIL.ImageDraw import ImageDraw
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

# from experiments.utils import save_img
from tensorflow.keras.preprocessing import image as image_keras_preprocessing
from PIL import Image
from tensorflow.keras.models import load_model
from time import time
import pygame as pg
import time


# for tensorflow detection
import cv2
import tflite
import zipfile
import tensorflow as tf
# tf.disable_v2_behavior()
#---


# loading keras models
if self_driving:
    print("Loading Keras: steering model")
    model = load_model('saved_model_custom_12.h5')
    print("Loading Keras: speed model")
    model2 = load_model('saved_model_road_1.h5')

    pg.init()

    screen = pg.display.set_mode((640, 480))
    screen_rect = screen.get_rect()
#---

# loading Tensorflow model
print("Loading Tensorflow: sign detection model")
MODEL_NAME = 'gym_duckietown/object_detection/inference_graph1'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/detect.tflite'
PATH_TO_LABELS = 'gym_duckietown/object_detection/inference_graph1/labelmap.pbtxt'

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = tf.lite.Interpreter(model_path=PATH_TO_FROZEN_GRAPH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
img_height = input_details[0]['shape'][1]
img_width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

#---

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        user_tile_start=(1.7, 1.6),
        accept_start_angle_deg=360,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

# Register a keyboard handler

key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

action = np.array([0.0, 0.0])

# make a config file !!!
max_speed = 0.9
max_angle = 0.9

angles = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
if self_driving:
    angles = 0
angle_idx = 2
color = (0, 200, 0)
angles_lst = []
speeds_lst = []
speed_folder = '2'
gl_speed = 0

sign_prediction = []
road_prediction = []
road_value = 3
road_value_prev = 3
sign_value = 0

speed_limit = 0.9
speed_limit_time = time.time()

in_intersection = False

# read directions
rd = open ("Directions.txt", "r")
directions_list = rd.readlines()
directions_list = [x.strip() for x in directions_list] 
print('DIRECTIONS:')
print(directions_list)
rd.close()
#---



def limit(speed, seconds):
    global speed_limit
    global speed_limit_time
    
    speed_limit = speed
    print('set speed limit', speed_limit)
    speed_limit_time = time.time() + seconds    

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    global action
    global max_speed
    global max_angle
    global angle_idx
    global angles
    global self_driving
    global speed_folder
    global gl_speed
    global color
    global angles_lst
    global speeds_lst
    global sign_prediction
    global sign_prediction
    global road_value
    global sign_value
    global road_value_prev
    global speed_limit
    global speed_limit_time
    global directions_list
    global in_intersection

    if key_handler[key.P]:
        angles = 0
        self_driving = True
    if key_handler[key.M]:
        angles = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
        self_driving = False
    if key_handler[key.UP]:
        action = np.add(action, np.array([0.05, 0]))
    if key_handler[key.DOWN]:
        action = np.add(action, np.array([-0.05, 0]))

    if not self_driving:
        if key_handler[key.LEFT]:
            angle_idx += 1
        if key_handler[key.RIGHT]:
            angle_idx -= 1
        if angle_idx < 0:
            angle_idx = 0
        if angle_idx > 6:
            angle_idx = 6

    if self_driving:
        action = np.array([action[0], angles])
    else:
        action = np.array([action[0], angles[angle_idx]])

    if action[0] > max_speed:
        action[0] = max_speed
    if action[0] < -max_speed:
        action[0] = -max_speed

    if not self_driving:
        if action[1] > max_angle:
            action[1] = max_angle
        if action[1] < -max_angle:
            action[1] = -max_angle
    print(' time.time()', time.time())
    print(' speed_limit_time',speed_limit_time)
    

    speed = action[0]
    angle = action[1]
    if self_driving:
        speed = gl_speed

    angle = speed / max_speed * angle
    angle *= 4.5

    act_cmd = np.array([round(speed, 4), round(angle, 4)])

    obs, reward, done, info = env.step(act_cmd)
    initial_img = Image.fromarray(obs)
    
    if self_driving:
        im = initial_img
        im = im.crop((0, 80, 640, 330))

        size = 66, 200
        im = im.resize(size, Image.ANTIALIAS)

        draw = ImageDraw(im)
        draw.rectangle((0, 185, 66, 200), fill=color)

        

        img_pred = im

        img_pred = image_keras_preprocessing.img_to_array(img_pred)
        img_pred = np.expand_dims(img_pred, axis=0)

        rslt = model.predict(img_pred)
        # print(rslt[0])
        speed = speed_limit
        rslt2 = model2.predict(img_pred)
        # print(rslt2[0])
        road_prediction.append(rslt2[0].argmax())
        if len(road_prediction) > 20:
            road_prediction.pop(0)
        print(road_prediction)
        rv = max(set(road_prediction), key = road_prediction.count)
        if rv != road_value_prev:
            road_value_prev = road_value
        road_value = rv
        if road_value == 3:
            sign_prediction = []
            sign_value = 0
        
        print('Road:' + str(road_value))
        speeds = [0.2, 0.4, 0.6, 0.8]
        gl_speed = 0
        for idx in range(4):
            gl_speed += rslt2[0][idx] * speeds[idx]
        
        if speed_limit_time < time.time():
            if len(speeds_lst) > 6:
                speeds_lst.pop(0)
            speeds_lst.append(round(gl_speed, 2))
            gl_speed = round(sum(speeds_lst) / len(speeds_lst), 2)
        else:
            print('///////////////STOP TIME////////////////')
            speeds_lst.pop(0)
            speeds_lst.append(speed_limit)
            gl_speed = round(sum(speeds_lst) / len(speeds_lst), 2)
            if gl_speed > 0:
                limit(0,4)
        
        print(speeds_lst)
        
        
            
        print(gl_speed)

        predicted_angles = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1, \
                            0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2,
                            0.3]
        final_angle = 0
        for idx in range(28):
            final_angle += rslt[0][idx] * predicted_angles[idx]
        # print('angle:' + str(final_angle))
        # angles = round(final_angle, 2)
        if len(angles_lst) > 2:
            angles_lst.pop(0)
        angles_lst.append(round(final_angle, 2))
        angles = round(sum(angles_lst) / len(angles_lst), 3)
        # print("Going with : " + str(angles))
        # print("Cache : " + str(angles_lst))

        # for idx in range(7):
        #     if angles[idx] == final_angle:
        #         angle_idx = idx







        image_width = 600
        image_height = 400

        print("----------------START-----------------")

        image = np.array(initial_img)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image, (img_width, img_height))
        input_data = np.expand_dims(image_resized, axis=0)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        for i, b in enumerate(boxes):
            if scores[i] >= 0.5:
                apx_dist = (0.2 * 3.04) / boxes[i][1]

        for i in range(len(scores)):
            if (scores[i] >= 0.7) and (scores[i] <= 1.0):
                ymin = int(max(1, boxes[i][0] * imH))
                xmin = int(max(1, boxes[i][1] * imW))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3]) * imW))

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                labelsize, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelsize[1] + 10)
                cv2.rectangle(image, (xmin, label_ymin - labelsize[1] - 10),
                              (xmin + labelsize[0], label_ymin + baseline - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                image_np = image

                if apx_dist < 0.7:
                    print(label)
                    if 'pedestrian_cross' in label:
                        sign_prediction.append(1)
                        cv2.putText(image_np, "CROSS", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)
                    if 'stop' in label:
                        sign_prediction.append(2)
                        cv2.putText(image_np, "STOP", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)
                    if 'priority' in label:
                        sign_prediction.append(3)
                        cv2.putText(image_np, "SLOW DOWN", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)
                    if len(sign_prediction) > 5:
                        sign_prediction.pop(0)
                    print(sign_prediction)
                    if len(sign_prediction) == 0:
                        sign_prediction.append(0)
                    sign_value = max(set(sign_prediction), key = sign_prediction.count)
                    print('Sign:' + str(sign_value))
    
        print("----------------FINISH-----------------")

        
        
        if sign_value == 2:  # stop sign
            print('prev',road_value_prev,'current',road_value)
            if road_value_prev == 3 and road_value == 0: # from straight road to intersection
                limit(0,4)
        if sign_value == 1:  # cross sign
            print('prev',road_value_prev,'current',road_value)
            if road_value_prev == 3 and road_value == 1: # from straight road to turning
                limit(0,4)

        if road_value_prev == 3 and road_value == 0: # from straight road to intersection
            new_direction = directions_list.pop(0)
            if new_direction == 'forward':
                color = (0, 200, 0)
            if new_direction == 'left':
                color = (200, 0, 0)
            if new_direction == 'right':
                color = (0, 0, 200)
            in_intersection = True
        
        if road_value == 3:
            color = (0, 200, 0)
            in_intersection = False

        if in_intersection == True:
            if color == (200, 0, 0):
                cv2.putText(image, 'Turning left', (50, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)
            if color == (0, 0, 200):
                cv2.putText(image, 'Turning right', (50, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)
            if color == (0, 200, 0):
                cv2.putText(image, 'Going forward', (50, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)

        pg_im = Image.fromarray(image)

        modepg = pg_im.mode
        sizepg = pg_im.size
        datapg = pg_im.tobytes()

        imagepg = pg.image.fromstring(datapg, sizepg, modepg)
        image_rect = imagepg.get_rect(center=screen.get_rect().center)
        screen.blit(imagepg, image_rect)
        pg.display.update()

        # get directions
        if key_handler[key.W]:
            color = (0, 200, 0)
        if key_handler[key.A]:
            color = (200, 0, 0)
        if key_handler[key.D]:
            color = (0, 0, 200)
        if key_handler[key.S]:
            color = (200, 200, 200)
        #---

    # save image start
    if key_handler[key.LCTRL] and self_driving is False:
        im = initial_img
        im = Image.fromarray(obs)

        
        im = im.crop((0, 80, 640, 330))

        global args
        folder_name = '0'
        if angles[angle_idx] > 0:
            folder_name = str(round(int(angles[angle_idx] * 10), 0))
        if angles[angle_idx] < 0:
            folder_name = '_' + str(round(int(angles[angle_idx] * -10), 0))

        size = 360, 120
        im = im.resize(size, Image.ANTIALIAS)

        img_name = 'classifier_road/{0}/{1}_{2}_{3}.png'.format(speed_folder,
                                                                str(round(time(), 1)),
                                                                str(round(env.cur_pos[0], 3)),
                                                                str(round(env.cur_pos[2], 3))
                                                                )
        im.save(img_name)
    # ---

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
