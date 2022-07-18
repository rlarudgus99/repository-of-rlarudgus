# -*- coding: utf-8 -*-
# -*- coding: cp949 -*-

import cv2
import numpy as np
import os
import natsort
import glob

path = 'C:/opencv_python/image_labeling_copy2'

files = natsort.natsorted(glob.glob(path+ '/*'))

ctr = 891

for f in files:
    os.rename(f,os.path.join(path,'Label_'+ str(ctr)+'.png'))
    ctr += 1

new_names = os.listdir(path)
sorted_new_names = natsort.natsorted(new_names)
print(sorted_new_names)
