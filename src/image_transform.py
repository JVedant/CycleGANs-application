import tensorflow as tf
from config import IMG_HEIGHT, IMG_WIDTH, CHANNELS

# to crop the image to our desired dimensions
def random_crop(image):
    image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, CHANNELS])
    return image


# to normalize the image to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


# generating images with random properties such as fliping the image on vertical axis
def random_jitter(image):
    image = tf.image.resize(
        image, [548, 548], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = random_crop(image)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_flip_left_right(image)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_flip_up_down(image)
    return image


# method to preprocess the train images
def load_train_image(image, label):
    image = random_jitter(image)
    image = normalize(image)
    return image


# method to preprocess the test images
def load_test_image(image, label):
    image = tf.image.resize(
        image, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = normalize(image)
    return image
