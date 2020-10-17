from PIL.ImageDraw import ImageDraw
from PIL import Image
import os
from os import listdir


#for direct in listdir('/home/dellen/Desktop/keras_sandbox_4/data/overtaking/train/'):
direct = '/home/dellen/Documents/bfmc/Virtual_Track/overtaking/trainx1/ee/'
for img in os.listdir(direct):
    print(img)
    im = Image.open(direct+img)
    draw = ImageDraw(im)
    draw.rectangle((0, 102, 140, 110), fill=(200, 0, 0))
    im.save(direct+img)
    #draw.rectangle((0, 61, 150, 66), fill=(0, 200, 0))
    #im.save(direct+img[:-4]+"_2.png")
    #draw.rectangle((0, 61, 150, 66), fill=(0, 0, 200))
    #im.save(direct+img[:-4]+"_3.png")
