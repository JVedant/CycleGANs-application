from layers import *
from config import IMG_WIDTH, IMG_HEIGHT, CHANNELS, weight_initializer, disc_x_optimizer

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input

# method to create the generator with arch (c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3)
# note: we are not compiling the generator here as the only job of it is to generate the images and learn from the discriminator's response
def generator():
    # declaring the shape of input image (here is it 256x256x3)
    gen_inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    gen = c7s1k(inputs=gen_inputs, k=64, activation=tf.nn.relu)
    gen = dk(inputs=gen, k=128)
    gen = dk(inputs=gen, k=256)
    
    for _ in range(9):
        gen = rk(inputs=gen, k=256)

    gen = uk(inputs=gen, k=128)
    gen = uk(inputs=gen, k=64)
    gen = c7s1k(inputs=gen, k=3, activation=tf.nn.tanh)
    model = Model(gen_inputs, gen)

    return model


# Method to create discriminator with the architecture mentioned in original paper (C64, C128, C256, C512)
def discriminator():

    # declaring the shape of input image (here is it 256x256x3)
    dis_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    
    # creating the arch
    d = CK(dis_input, 64, use_instancenorm=False)
    d = CK(d, 128)
    d = CK(d, 256)
    d = CK(d, 512)

    # layer to classify between the originality of generated images
    d = Conv2D(1, kernel_size=(4,4), kernel_initializer=weight_initializer, padding='same')(d)

    # generating a model using inputs and outputs
    dis = Model(dis_input, d)

    # compiling the model as it would be used to classify the images
    dis.compile(loss='mse', optimizer=disc_x_optimizer)
    return dis
    # can change the loss function to binarycrossentropy and adding a Fully connected network as ouput layer (have to experiment)