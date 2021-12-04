import tensorflow as tf
from models import generator, discriminator
from config import *
from loss import *
from image_transform import load_train_image, load_test_image
from tensorflow.train import Checkpoint
import tensorflow_datasets as tfds
from time import time
from utils import generate_images
from tqdm import tqdm

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

# Defining the generators and discriminators
generator_g = generator_f = generator()
discriminator_x = discriminator_y = discriminator()


@tf.function
def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = cycle_loss(real_x, cycled_x) + cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(
        total_gen_g_loss, generator_g.trainable_variables
    )
    generator_f_gradients = tape.gradient(
        total_gen_f_loss, generator_f.trainable_variables
    )
    discriminator_x_gradients = tape.gradient(
        disc_x_loss, discriminator_x.trainable_variables
    )
    discriminator_y_gradients = tape.gradient(
        disc_y_loss, discriminator_y.trainable_variables
    )

    # Apply the gradients to the optimizer
    gen_g_optimizer.apply_gradients(
        zip(generator_g_gradients, generator_g.trainable_variables)
    )
    gen_f_optimizer.apply_gradients(
        zip(generator_f_gradients, generator_f.trainable_variables)
    )
    disc_x_optimizer.apply_gradients(
        zip(discriminator_x_gradients, discriminator_x.trainable_variables)
    )
    disc_y_optimizer.apply_gradients(
        zip(discriminator_y_gradients, discriminator_y.trainable_variables)
    )


def run():
    # loading the dataset from pre defined tensorflow_dataset module (monet2photo)
    data = tfds.load("cycle_gan/maps", as_supervised=True, data_dir="../DATA")
    # dataset is split as ...
    #'testA' 	121
    #'testB' 	751
    #'trainA' 	1,072
    #'trainB' 	6,287

    # assigning the monet to x and photo to y
    train_y, train_x, test_y, test_x = (
        data["trainA"],
        data["trainB"],
        data["testA"],
        data["testB"],
    )

    # preprocessing all the images using above mentioned functions along with shuffeling the data and converting in specific batch sizes
    train_x = train_x.map(load_train_image, num_parallel_calls=AUTOTUNE)
    train_x = train_x.shuffle(buffer_size=BUFFER_SIZE)
    train_x = train_x.batch(BATCH_SIZE)
    train_x = train_x.prefetch(buffer_size=AUTOTUNE)

    train_y = train_y.map(load_train_image, num_parallel_calls=AUTOTUNE)
    train_y = train_y.shuffle(buffer_size=BUFFER_SIZE)
    train_y = train_y.batch(BATCH_SIZE)
    train_y = train_y.prefetch(buffer_size=AUTOTUNE)

    test_x = test_x.map(load_test_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    test_y = test_y.map(load_test_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

    sample_x = next(iter(test_x))
    sample_y = next(iter(test_y))

    # Declaring A Checkpoint so that we can start the training from where we stopped last time as we don't want to train from the scratch everytime

    # path to store the checkpoints
    # checkpoint_path = "checkpoints_monet2photo/train"
    checkpoint_path = "../Logs/checkpoints/train"

    ckpt = Checkpoint(
        generator_g=generator_g,
        generator_f=generator_f,
        discriminator_x=discriminator_x,
        discriminator_y=discriminator_y,
        gen_g_optimizer=gen_g_optimizer,
        gen_f_optimizer=gen_f_optimizer,
        disc_x_optimizer=disc_x_optimizer,
        disc_y_optimizer=disc_y_optimizer,
    )

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!!!")
        print("*" * 50)

    # Loop to train the GAN
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        start = time()

        for x, y in tqdm(
            tf.data.Dataset.zip((train_x, train_y)), desc="Dataset iterator", total=1092
        ):
            train_step(x, y)

        generate_images(generator_g, sample_x, epoch)

        # defining the interval to save checkpoints
        if (epoch + 1) % 3 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(
                "Saving checkpoint for epoch {} at {}".format(epoch + 1, ckpt_save_path)
            )

        print("Time taken for epoch {} is {} sec\n".format(epoch + 1, time() - start))

    generator_f.save("../models/generator_f.h5")
    generator_g.save("../models/generator_g.h5")
    discriminator_x.save("../models/discriminator_x.h5")
    discriminator_y.save("../models/discriminator_y.h5")


if __name__ == "__main__":
    run()
