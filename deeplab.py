import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import numpy as np

from utils import layer_dict, process_numpy, show, ShowCallBack, CustomIoU


img_data = np.load("./seg_dataset/img128.npy")
seg_data = np.load("./seg_dataset/seg128.npy")


def get_model(img_shape=(128, 128, 3), filter_num=16):
    inputs = keras.layers.Input(shape=(img_shape))
    o = inputs
    for l, s in reversed(layer_dict.items()):
        if l == img_shape[0]:
            o = keras.layers.Conv2D(filters=filter_num * s, kernel_size=3, strides=1, padding="SAME")(o)
            o = keras.layers.BatchNormalization()(o)
            o = keras.layers.LeakyReLU(0.2)(o)
        if l < img_shape[0]:
            o = keras.layers.Conv2D(filters=filter_num * s, kernel_size=3, strides=2, padding="SAME")(o)
            o = keras.layers.BatchNormalization()(o)
            o = keras.layers.LeakyReLU(0.2)(o)

    o = keras.layers.Conv2D(filters=filter_num * layer_dict[o.shape[1]], kernel_size=3, strides=1, dilation_rate=2, padding="SAME")(o)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.LeakyReLU(0.2)(o)

    pool = keras.layers.AveragePooling2D(pool_size=img_shape[0]//o.shape[1])(inputs)

    c1 = keras.layers.Conv2D(filters=filter_num * layer_dict[o.shape[1]], kernel_size=1, strides=1, padding="SAME")(o)
    c1 = keras.layers.BatchNormalization()(c1)
    c1 = keras.layers.LeakyReLU(0.2)(c1)

    c6 = keras.layers.Conv2D(filters=filter_num * layer_dict[o.shape[1]], kernel_size=3, strides=1, dilation_rate=6, padding="SAME")(o)
    c6 = keras.layers.BatchNormalization()(c6)
    c6 = keras.layers.LeakyReLU(0.2)(c6)

    c12 = keras.layers.Conv2D(filters=filter_num * layer_dict[o.shape[1]], kernel_size=3, strides=1, dilation_rate=12, padding="SAME")(o)
    c12 = keras.layers.BatchNormalization()(c12)
    c12 = keras.layers.LeakyReLU(0.2)(c12)

    c18 = keras.layers.Conv2D(filters=filter_num * layer_dict[o.shape[1]], kernel_size=3, strides=1, dilation_rate=18, padding="SAME")(o)
    c18 = keras.layers.BatchNormalization()(c18)
    c18 = keras.layers.LeakyReLU(0.2)(c18)

    o = keras.layers.Concatenate(axis=-1)([pool,c1,c6,c12,c18])
    o = keras.layers.Conv2D(filters=filter_num * layer_dict[o.shape[1]], kernel_size=1, strides=1, padding="SAME")(o)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.LeakyReLU(0.2)(o)

    for l, s in layer_dict.items():
        if l == img_shape[0] // 2:
            break
        o = keras.layers.Conv2DTranspose(filters=filter_num * layer_dict[o.shape[1]], kernel_size=1, strides=2, padding="SAME")(o)
        o = keras.layers.BatchNormalization()(o)
        o = keras.layers.LeakyReLU(0.2)(o)

    o = keras.layers.Conv2DTranspose(filters=256, kernel_size=1, strides=2, padding="SAME")(o)
    return keras.Model(inputs=inputs, outputs=o)


dataset = process_numpy(img_data, seg_data, batch_size=8)
model = get_model()
model.summary()
keras.utils.plot_model(model, show_shapes=True)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=CustomIoU(num_classes=256))
model.fit(dataset, epochs=20)
show(seg_data[:16], model(img_data[:16]))
