import tensorflow as tf
from config import loss_fn, LAMBDA

# Methods to find the error (loss) in data
# This method is used to find how much does the generated image differ than the original 
def discriminator_loss(real, generated):
    real_loss = loss_fn(tf.ones_like(real), real)
    generated_loss = loss_fn(tf.zeros_like(generated), generated)
    total_loss = real_loss + generated_loss
    return total_loss * 0.5


# This method is used to find the difference between generated images to true +ve 
def generator_loss(generated):
    return loss_fn(tf.ones_like(generated), generated)


# It is the main component in CycleGAN which finds the difference between the cycled images i.e. the image generated again from the regenerated image to find out whether the input image is again generated when used in backward
def cycle_loss(real_image, cycled_image):
    loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss


# used to prevent the loss in colors and fine details
def identity_loss(real_image, sample_image):
    loss = tf.reduce_mean(tf.abs(real_image - sample_image))
    return LAMBDA * loss * 0.5