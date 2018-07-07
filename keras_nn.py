import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler
from Data import FDDB_Data
from keras.models import Sequential
from keras import losses, optimizers, utils, regularizers
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dense, Dropout
import matplotlib.pylab as plt

# def scale_data(data):
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(data)
#     return scaled_data

def build_cnn(model, input_shape, num_classes, reg):

    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=input_shape, padding='same', kernel_regularizer=regularizers.l1(reg)))
    model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.01))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l1(reg)))
    model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.01))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l1(reg)))
    model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.01))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1500, activation='relu', kernel_regularizer=regularizers.l1(reg)))
    model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.l1(reg)))
    model.add(Dense(num_classes, activation='softmax'))

class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

batch_size = 128
num_classes = 2
epochs = 1

data = FDDB_Data()
(pos_vector_space, neg_vector_space, train_y) = data.load(img_size=[60,60], n_samples = 5000, train=True)
train_x = np.append(pos_vector_space, neg_vector_space, axis=0)
train_x = train_x/255
input = np.append(train_x, train_y[:,None], axis=1)
np.random.shuffle(input)
np.random.shuffle(input)
train_x = input[:,0:3600]
train_y = input[:,3600]

(pos_test_images, neg_test_images, test_y) = data.load(img_size = [60,60], n_samples=100, train=False)
test_x = np.append(pos_test_images, neg_test_images, axis=0)
test_x = test_x/255
train_x = train_x.reshape(-1, 60 , 60, 1)
test_x = test_x.reshape(-1, 60, 60, 1)

train_y = utils.to_categorical(train_y, num_classes)
test_y = utils.to_categorical(test_y, num_classes)

max_count=10
lambdaa=np.zeros(max_count)
epsilonn=np.zeros(max_count)
score_acc=np.zeros(max_count)
for count in range(max_count):

    history = AccuracyHistory()

    model = Sequential()
    lambdaa[count] = 10**np.random.uniform(-5,-3)
    epsilonn[count] = 10**np.random.uniform(-3,-4)
    build_cnn(model, input_shape = (60,60,1), num_classes=2, reg=lambdaa[count])
    model.compile(loss=losses.categorical_crossentropy,
                optimizer=optimizers.Adam(lr=epsilonn[count]),
                metrics=['accuracy'])
    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, history]

    model.fit(train_x, train_y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(test_x, test_y),
            callbacks=callbacks_list)

    score = model.evaluate(test_x, test_y, verbose=1)
    score_acc[count] = score[1]
#    print('Test loss: {}, Test accuracy: {}, lr: {}, reg: {}'.format(score[0], score[1], lambdaa, epsilonn))

# plt.plot(range(1, epochs+1), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()
for i in range (max_count):
    print('No: ', i, 'Regularizer: ', lambdaa[i], 'Learning rate: ', epsilonn[i], 'Accuracy: ', score_acc[i] )

ind = np.unravel_index(np.argmax(score_acc, axis = None), score_acc.shape)
print( 'Regularizer: ', lambdaa[ind], 'Learning rate: ', epsilonn[ind], 'Accuracy: ', score_acc[ind] )
