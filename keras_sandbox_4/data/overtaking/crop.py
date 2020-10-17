from PIL import Image
import os
from os import listdir

size = 150, 66
directs = ["wa","wb","wc","wd","we","wf","wg"]
for direct in listdir('/home/dellen/Desktop/keras_sandbox_4/data/overtaking/validation/').append('/home/dellen/Desktop/keras_sandbox_4/data/overtaking/train/'):  
        direct = '/home/dellen/Desktop/keras_sandbox_4/data/overtaking/validation/' + str(direct)+'/'
        for img in os.listdir(direct):
                print(img)
                #im = Image.open(direct+img)
                #im_resized = im.crop((60, 28, 300, 120))
                #im_resized = im.resize(size, Image.ANTIALIAS)
                #im_resized.save(direct+img)
