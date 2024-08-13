#%% Import library
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

#%% define generator
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, 
                             zoom_range=0.2, horizontal_flip=True)

#%% load file
path = 'C:/Users/Ricky/Desktop/大三下 課程/機器學習導論/HW5/Animals-1'
os.chdir(path)
filelist = os.listdir(path)
filelist.sort(key=lambda x: int(x.split('.')[0]))
fignum = int(len(filelist)/100)
image = np.zeros([fignum,64,64,4])  # 200,200,3 = 200*200 rgb data
for i in range(fignum):
    image[i,:,:,:] = matplotlib.image.imread(filelist[i])

#%% create pictures
datanum = 0
savepath = 'C:/Users/Ricky/Desktop/Animals-3/'
for data in datagen.flow(image, save_to_dir=savepath, save_prefix='gen'):
    image = np.append(image, data, axis=0)
    datanum = datanum + 1
    if datanum > 3:
        break
