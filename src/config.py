import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

AUTOTUNE = tf.data.experimental.AUTOTUNE

# used to prevent identity loss i.e. loss in the color of picture when transforming
LAMBDA = 10

# number of iteration to perform the training
EPOCHS = 100

# used to shuffle 1000 images
BUFFER_SIZE = 100

# declaring batches to train on a number of specified images simultaneously
BATCH_SIZE = 1

# image dimensions
IMG_WIDTH = 512
IMG_HEIGHT = 512
CHANNELS = 3

# setting random normalized weights
weight_initializer = RandomNormal(stddev=0.02)

# the loss function used here is BinaryCrossentropy as we have to classify whether the generated image matches the original or not
loss_fn = BinaryCrossentropy(from_logits=True)

# the optimizers are set to Adam with learning_rate as 2e-4 and beta_1 as 0.5
gen_g_optimizer = gen_f_optimizer = disc_x_optimizer = disc_y_optimizer = Adam(
    learning_rate=3e-4, beta_1=0.5
)
