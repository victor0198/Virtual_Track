#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
self_driving = False

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

from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.models import load_model
from time import time
import pygame as pg


# keras model
if self_driving:
    print("!!!------ LOADING Keras ------!!!")
    #
    # import tensorflow.keras
    # from PIL import Image, ImageOps
    # import numpy as np
    #
    # # Disable scientific notation for clarity
    # np.set_printoptions(suppress=True)
    #
    # # Load the model
    # model = tensorflow.keras.models.load_model('keras_model0.h5')
    #
    # # Create the array of the right shape to feed into the keras model
    # # The 'length' or number of images you can put into the array is
    # # determined by the first position in the shape tuple, in this case 1.
    # data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    #
    #

    # load model
    model = load_model('saved_model_custom_12.h5')
    # summarize model.
    model.summary()
    # end keras

    pg.init()

    screen = pg.display.set_mode((640, 480))
    screen_rect = screen.get_rect()

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

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)


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

    # print("ACTION:" + str(action))
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
            # action = np.add(action, np.array([0, 0.3]))
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
    # action = np.add(action, np.array([0, -0.3]))
    # if key_handler[key.SPACE]:
    #     action[0] = action[0]/1.2
    # action[1] = action[1]/1.05

    # Speed boost
    # if key_handler[key.LSHIFT]:
    #     action *= 1.2

    if action[0] > max_speed:
        action[0] = max_speed
    if action[0] < -max_speed:
        action[0] = -max_speed

    if not self_driving:
        if action[1] > max_angle:
            action[1] = max_angle
        if action[1] < -max_angle:
            action[1] = -max_angle

    speed = action[0]
    angle = action[1]

    # if speed < 0.03 and speed > -0.03:
    #     angle = 0
    print("GOING WITH ANGLE:" + str(angle))
    angle = speed / max_speed * angle
    angle *= 4.5

    act_cmd = np.array([round(speed, 4), round(angle, 4)])

    obs, reward, done, info = env.step(act_cmd)
    # print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    global color
    global angles_lst
    if self_driving:
        im = Image.fromarray(obs)
        im = im.crop((0, 80, 640, 330))

        size = 66, 200
        im = im.resize(size, Image.ANTIALIAS)

        draw = ImageDraw(im)
        draw.rectangle((0, 185, 66, 200), fill=color)

        modepg = im.mode
        sizepg = im.size
        datapg = im.tobytes()

        imagepg = pg.image.fromstring(datapg, sizepg, modepg)
        image_rect = imagepg.get_rect(center=screen.get_rect().center)
        screen.blit(imagepg, image_rect)
        pg.display.update()

        img_pred = im

        img_pred = image.img_to_array(img_pred)
        img_pred = np.expand_dims(img_pred, axis=0)

        rslt = model.predict(img_pred)

        print(rslt[0])

        predicted_angles = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2,
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

    if key_handler[key.W]:
        color = (0, 200, 0)
    if key_handler[key.A]:
        color = (200, 0, 0)
    if key_handler[key.D]:
        color = (0, 0, 200)
    if key_handler[key.S]:
        color = (200, 200, 200)

    if key_handler[key.LCTRL] and self_driving is False:

        im = Image.fromarray(obs)

        # !!! save image
        # # print(im.size)
        im = im.crop((0, 80, 640, 330))

        global args
        folder_name = '0'
        if angles[angle_idx] > 0:
            folder_name = str(round(int(angles[angle_idx] * 10), 0))
        if angles[angle_idx] < 0:
            folder_name = '_' + str(round(int(angles[angle_idx] * -10), 0))

        size = 360, 120
        im = im.resize(size, Image.ANTIALIAS)
        draw = ImageDraw(im)
        if color == (200, 200, 200):
            # print('all')
            for colo in [(0, 0, 200), (0, 200, 0), (200, 0, 0)]:
                # print(colo)

                draw.rectangle((0, 115, 360, 120), fill=colo)
                # im.show()
                img_name = 'classifier5/{0}/{1}/{2}_{3}_{4}_{5}.png'.format('left',
                                                                            folder_name,
                                                                            str(round(time(), 1)),
                                                                            str(round(env.cur_pos[0], 3)),
                                                                            str(round(env.cur_pos[2], 3)),
                                                                            str(int(
                                                                                colo[0] / 200 + colo[1] / 100 + colo[
                                                                                    2] / 66.66))
                                                                            )
                im.save(img_name)
        else:
            draw.rectangle((0, 115, 360, 120), fill=color)

            img_name = 'classifier5/{0}/{1}/{2}_{3}_{4}.png'.format('left',
                                                                    folder_name,
                                                                    str(round(time(), 1)),
                                                                    str(round(env.cur_pos[0], 3)),
                                                                    str(round(env.cur_pos[2], 3))
                                                                    )
            im.save(img_name)
        # end save image

        # !!! keras prediction
        # Replace this with the path to your image
        # image = im # Image.open('test_photo.jpg')
        #
        # # resize the image to a 224x224 with the same strategy as in TM2:
        # # resizing the image to be at least 224x224 and then cropping from the center
        # size = (224, 224)
        # image = ImageOps.fit(image, size, Image.ANTIALIAS)
        #
        # # turn the image into a numpy array
        # image_array = np.asarray(image)
        #
        # # display the resized image
        # # image.show()
        #
        # # Normalize the image
        # normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        #
        # # Load the image into the array
        # data[0] = normalized_image_array
        #
        # # run the inference
        # prediction = model.predict(data)
        # print(prediction)
        # end keras

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
