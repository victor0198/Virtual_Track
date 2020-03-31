from PIL.ImageDraw import ImageDraw
from PIL import Image
import os

directs = ['aa/','ab/','ac/','ad/','ae/','af/','ag/']
for direct in directs:
        direct = 'train/' + direct
        for img in os.listdir(direct):
                im = Image.open(direct+img)
                draw = ImageDraw(im)
                draw.rectangle((0, 61, 200, 66), fill=(150, 150, 150))
                im.save(direct+img)
