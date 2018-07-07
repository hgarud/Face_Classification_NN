import numpy as np
import cv2
import os

image_out_dir = "./Data/augmented_gans/"
# image_out_dir = "Data/more_negImages/"
img_count = 0
# pos_im_dir = "./DCGAN-tensorflow/samples_"+dir+"_64BS/"
# pos_im_dir = "./DCGAN-tensorflow/samples_1_128BS/"
pos_im_dir = "./DCGAN-tensorflow/training_samples/train"+dir+"/"
pos_images = os.listdir(pos_im_dir)
print(len(pos_images))
for pos_image in pos_images:
    batch_image = cv2.imread(pos_im_dir+pos_image)
    for i in range(0,batch_image.shape[0],60):
        for j in range(0,batch_image.shape[1],60):
            face = batch_image[i:i+60,j:j+60]
            img_count = img_count+1
            print(img_count)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(image_out_dir+str(img_count)+".png", face.astype('uint8'))
