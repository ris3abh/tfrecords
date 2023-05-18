import tensorflow as tf
from tensorflow.keras.applications import InceptionV3

def birder_parse_example(serialized_examples):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'birdType': tf.io.FixedLenFeature([], tf.int64)
    }
    examples = tf.io.parse_example(serialized_examples, feature_description)
    targets = examples.pop('birdType')
    images = tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(examples['image'], channels=3), tf.float32), 299, 299)
    return images, targets

tf_train = tf.data.TFRecordDataset(['../data/birds-10-eachOf-358.tfrecords'])
tf_validation = tf.data.TFRecordDataset(['../data/birds-10-eachOf-358.tfrecords'])

train_dataset = tf_train.map(birder_parse_example, num_parallel_calls=2)
validation_dataset = tf_validation.map(birder_parse_example, num_parallel_calls=2)

train_dataset = train_dataset.map(lambda x, y: (tf.keras.applications.inception_v3.preprocess_input(x), y)).batch(32)
validation_dataset = validation_dataset.map(lambda x, y: (tf.keras.applications.inception_v3.preprocess_input(x), y)).batch(32)
print("datasets defined....")

base_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)
print("base model defined....")

avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(358, activation="softmax")(avg)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable=False

initial_epochs = 10

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('birder', save_best_only=True)
earlyStop_cb = tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
ss=5e-1
optimizer = tf.keras.optimizers.SGD(learning_rate=ss)
model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer, metrics=["accuracy"])
print("model compiled....")

print("training started....")
model.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs, callbacks=[checkpoint_cb,earlyStop_cb])

base_model.trainable = True

fine_tune_learning_rate = 1e-5

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_epochs = 5
history = model.fit(train_dataset, epochs=initial_epochs + fine_tune_epochs, validation_data=validation_dataset, callbacks=[earlyStop_cb])


model.save('../model/buildAndTrainBirder.h5')

import zipfile
with zipfile.ZipFile('../model/buildAndTrainBirder.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('../model/buildAndTrainBirder.h5')

print("model saved....")