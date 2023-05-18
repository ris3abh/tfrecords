import tensorflow as tf
from tensorflow.keras.applications import InceptionV3

nToAugment=4
from tensorflow import keras

def augmentImages(image, label):
    resized_image = tf.image.resize_with_pad(image, 224, 224)
    image_list = [resized_image]
    augmented_images = []
    for _ in range(nToAugment):
        # Apply random transformations
        augmented_image = tf.image.random_flip_left_right(resized_image)
        augmented_image = tf.image.random_flip_up_down(augmented_image)
        augmented_image = tf.image.random_brightness(augmented_image, max_delta=0.4)
        augmented_image = tf.image.random_contrast(augmented_image, lower=0.6, upper=1.4)
        augmented_images.append(augmented_image)
    labels = [label for _ in range(nToAugment + 1)]
    image_list.extend(augmented_images)
    return image_list, labels

def augment_images(image, label):
    image_list, label_list = augmentImages(image, label)
    return image_list, label_list

