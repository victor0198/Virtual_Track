from PIL import Image
import os
from os import listdir

size = 200, 66
for direct in listdir('/home/dellen/Documents/bfmc/Virtual_Track/keras_sandbox_4/data/validation/'):
        direct = 'validation/' + str(direct)+'/'
        for img in os.listdir(direct):
                im = Image.open(direct+img)
                im_resized = im.resize(size, Image.ANTIALIAS)
                im_resized.save(direct+img)
