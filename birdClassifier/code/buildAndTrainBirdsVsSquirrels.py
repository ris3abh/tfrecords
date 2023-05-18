import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping
from preprocessDefiniton import augment_images

train_dataset_path = '../data/birds-vs-squirrels-train.tfrecords'
val_dataset_path = '../data/birds-vs-squirrels-validation.tfrecords'

image_width, image_height = 224, 224
batch_size = 16

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

num_classes = 3

def parse_tfrecord(example):
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.image.resize(image, [image_width, image_height])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(example['label'], num_classes)
    return image, label

train_dataset = tf.data.TFRecordDataset(train_dataset_path)
train_dataset = train_dataset.map(parse_tfrecord)
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.map(augment_images)
train_dataset = train_dataset.unbatch()
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.TFRecordDataset(val_dataset_path)
val_dataset = val_dataset.map(parse_tfrecord)
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))
base_model.trainable = False
print("base model defined....")

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("model compiled....")

initial_epochs = 10

early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
print("training model....")
model.fit(train_dataset, epochs=initial_epochs, validation_data=val_dataset, callbacks=[early_stopping])

base_model.trainable = True

fine_tune_learning_rate = 1e-5

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_epochs = 5

history = model.fit(train_dataset, epochs=initial_epochs+fine_tune_epochs, validation_data=val_dataset, callbacks=[early_stopping])

model.save('../modelsbirdsVsSquirrelsModel.h5')

import tarfile
print("converting .h5 to tgz file....")
with tarfile.open('../modelsbirdsVsSquirrelsModel.tgz', 'w:gz') as f:
    f.add('../modelsbirdsVsSquirrelsModel.h5')

print("tgz file created....")

print("model trained.... printing graphs....")
## loss and accuracy
import matplotlib.pyplot as plt

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

epoch_range = range(1, len(training_loss) + 1)
plt.plot(epoch_range, training_loss)
plt.plot(epoch_range, val_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

plt.plot(epoch_range, training_accuracy)
plt.plot(epoch_range, val_accuracy)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()


