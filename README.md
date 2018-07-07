# Face_Classification_NN
A neural network extension of the Face Classification [repository].

## Requirements
* Python 3.6.2
* OpenCV 3.3.0
* Tensorflow (GPU version)
* Keras


Any type of neural network typically requires a large amount of training data for effective inference. Hence, there is a need to increase the size of the dataset used previously.

Steps that can be taken to increase the existing dataset:
* Go to [this link] and select one of the provided 17 face datasets which has face bounding boxes annotated. Download the dataset (note that some of the 17 datasets might need registration to download and you can skip those to save time).  
Extract n = 10,000 training images (or more, you can combine multiple datasets) for face and non-face respectively.
* Augment the existing dataset using image transformations like random crop, mirroring, adding random noise, etc.
* Be fancy! Train a pair of adversarial networks to generate images for you. (That's what I did :p)  
Use **​Deep Convolutional ​Generative Adversarial Networks (DCGAN)** to augment the
dataset. I implemented the Tensorflow implementation of a DCGAN from [carpedm20's repo].

## Data Augmentation

Fantastic papers [here] and [there] on DCGAN and GANs. Read.

Implementation of DCGAN is straight-forward and according to the README provided in the above linked repo. Using the images from FDDB dataset as training images for the networks, I was able to generate images having socially acceptable faces.  
Using a **batch size of 64** (increasing the batch size didn't help. Did not tune other parameters as this was just an exploratory excursion into GANs), the networks gave me the following images:  
*(Reader beware! Parental guidance advised.)*

* After 100 epochs *( I have had nightmares from images produced until this epoch :( )*

![GAN 100eps][100eps]

* After 200 epochs *( I did not have nightmares from images produced at this epoch :) )*

![GAN 200eps][200eps]

* After 225 epochs *( I have had nightmares from images produced after this epoch :( )*

![GAN 225eps][225eps]

Run the ***augment_data.py*** to extract augmented face images from the DCGAN output. *Be sure to change out the directory paths in the script.*

## Data Preprocessing
There are two main methods of preprocessing training data for neural nets/deep neural nets:
1. Zero - centered and normalized data
2. Normalized data

  #### Zero - centered and normalized data
  Subtracting each feature with its mean and subtracting by its standard deviation normalizes the features to be in the range of -1 and 1 which speeds up convergence. This is achieved by the ​**sklearn.preprocessing.StandardScaler** library in Python.  
![Zero-centered data][Prep]

  #### Normalized data
  To scale the data between 0 and 1, we divide the data by the maximum possible value, i.e. 255.
  Helps in using ReLU as the activation function as the range after preprocessing is between 0 and 1.

## Network Architecture
**3 Convolutional layers** ​with ​**32**, **​64**​, and **​128** ​filters ​respectively.
This is followed by a ​**dense neural network** ​having **​2 hidden layers**​ with ​**1500** and **​200 neurons** respectively. The output of this network is a **​one-hot encoded** ​vector with 2 classes for Face and Non-Face class.

***keras_nn.py*** uses the Keras higher level API with a GPU enabled Tensorflow backend.  
***tf_cnn.py*** uses the native TEnsorflow API.

## Findings
Using the **SGD optimizer saturated the test accuracy to 97% for 2000 training samples**.
Increasing the number of training samples provided no improvement whatsoever. In addition to
that, the time required to converge on this accuracy is more for SGD - around the **18th epoch**. Plus, the training was noisy, even with reduced learning rate.
Switching to Adam optimizer boosted the overall performance of the network in the following
ways:
* Firstly, the maximum accuracy achieved was **99%**.
* Secondly, **faster convergence** to the maximum accuracy - around the **5th epoch**.
* However, Adam optimizer **required a larger training data-set**. In test runs the size of the training data-set was 10,000 samples - 5000 samples for each of
the two classes.
* Thirdly, much smoother training with **L1 regularization(Lambda=1e-4)**.
If we perform a ​grid search ​ over the values of regularization and learning rate, we find
optimized parameters to get the best results. However, with the limited hardware, the search is
only possible for ​1 epoch ​ and for ​10 distinct values.

**Adam optimizer pros:**
* Smoother (better?) training.
* Faster convergence.
* Greater test accuracy.

**Adam optimizer cons:**
* Requires larger data set.

[repository]: https://github.com/hgarud/Face_Classification
[this link]: https://github.com/betars/Face-Resources
[carpedm20's repo]: https://github.com/carpedm20/DCGAN-tensorflow
[here]: https://arxiv.org/pdf/1511.06434.pdf
[there]: https://arxiv.org/pdf/1406.2661.pdf
[prep]: https://github.com/hgarud/Face_Classification_NN/blob/master/Graphics/Zero-centered.png
[100eps]: https://github.com/hgarud/Face_Classification_NN/blob/master/Graphics/GAN-Faces-100eps.png
[200eps]: https://github.com/hgarud/Face_Classification_NN/blob/master/Graphics/GAN-Faces-200eps.png
[225eps]: https://github.com/hgarud/Face_Classification_NN/blob/master/Graphics/GAN-Faces-225eps.png
