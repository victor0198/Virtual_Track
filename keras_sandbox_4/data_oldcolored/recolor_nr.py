from PIL.ImageDraw import ImageDraw
from PIL import Image
import os

directs = ['aa/','ab/','ac/','ad/','ae/','af/','ag/']
for direct in directs:
	for img in os.listdir(direct):
                i_l = img.split('.')  
                type_i = int(i_l[3][len(i_l[3])-1])
                if type_i==1:
                    color = (150, 10, 10)
                elif type_i==2:
                    color = (10, 150, 10)
                elif type_i==3:
                    color = (10, 10, 150)
                else:
                    print(img + '--------------')
                im = Image.open(direct+img)
                
                draw = ImageDraw(im)
                draw.rectangle((0, 61, 200, 66), fill=color)
                im.save(direct+img)
