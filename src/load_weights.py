import tensorflow as tf
import numpy as np
import emotion_gender_age_model as ega
image_size = 128
n_classes={'emotion':7,'age':0,'gender':2}
image_shape = (image_size, image_size, 3)
image_input = tf.keras.layers.Input(shape=image_shape)


# model = ega.create_model(image_input, n_classes=n_classes)
# model = ega.set_non_trainable(model,block_ids=['bolck9'])
model = ega.load_model("checkpoints/run1/EGA_epoch_19_score_73.model")
fake_data = np.ones(shape=[12, image_size,image_size, 3]).astype(np.float32)
model(fake_data) # initialize model to load weights
model.load_weights("./checkpoints/run1/EGA_epoch_19_score_73.h5")