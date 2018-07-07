import cv2
import os
import numpy as np
# from batchup import data_source

class FDDB_Data:
    def __init__(self):
        # self.pos_im_dir = "./Data/posImages/"
        self.pos_im_dir = "./Data/augmented_10k/"
        self.neg_im_dir = "./Data/negImages/"
        # self.neg_im_dir = "./Data/final_non_images_1/"
        self.pos_test_im_dir = "./Data/testImages/pos/"
        self.neg_test_im_dir = "./Data/testImages/neg/"

    def load(self, train, n_samples=1000, img_size=[10,10]):
        if train:
            pos_images = os.listdir(self.pos_im_dir)
            neg_images = os.listdir(self.neg_im_dir)
            count = 0
            pos_vector_space = []
            for pos_image in pos_images[0:n_samples]:
                image = cv2.imread(self.pos_im_dir+pos_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (img_size[0],img_size[1]))
                im_vector = image.flatten()
                pos_vector_space.append(im_vector)
            pos_vector_space = np.array(pos_vector_space)

            neg_vector_space = []
            for neg_image in neg_images[0:n_samples]:
                image = cv2.imread(self.neg_im_dir+neg_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (img_size[0],img_size[1]))
                im_vector = image.flatten()
                neg_vector_space.append(im_vector)
            neg_vector_space = np.array(neg_vector_space)

            # labels = np.append(np.ones(len(pos_vector_space)), np.zeros(len(neg_vector_space)), axis = 1)
            pos_labels = np.ones(len(pos_vector_space))
            neg_labels = np.zeros(len(neg_vector_space))
            labels = np.append(pos_labels, neg_labels, axis = 0)

            return pos_vector_space, neg_vector_space, labels
        else:
            pos_test_images = os.listdir(self.pos_test_im_dir)
            neg_test_images = os.listdir(self.neg_test_im_dir)
            pos_test_vector_space = []
            for pos_test_image in pos_test_images[0:n_samples]:
                image = cv2.imread(self.pos_test_im_dir+pos_test_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (img_size[0],img_size[1]))
                im_vector = image.flatten()
                pos_test_vector_space.append(im_vector)
            pos_test_vector_space = np.array(pos_test_vector_space)

            neg_test_vector_space = []
            for neg_test_image in neg_test_images[0:n_samples]:
                image = cv2.imread(self.neg_test_im_dir+neg_test_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (img_size[0],img_size[1]))
                im_vector = image.flatten()
                neg_test_vector_space.append(im_vector)
            neg_test_vector_space = np.array(neg_test_vector_space)

            pos_labels = np.ones(len(pos_test_vector_space))
            neg_labels = np.zeros(len(neg_test_vector_space))
            labels = np.append(pos_labels, neg_labels, axis = 0)

            return pos_test_vector_space, neg_test_vector_space, labels
