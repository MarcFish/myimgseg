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
    m = keras.applications.ResNet50(include_top=False, weights=None, input_shape=img_shape)
    o = m(o)
    org_o = keras.layers.Conv2D(filters=filter_num * layer_dict[o.shape[1]], kernel_size=1, strides=1)(o)
    org_o = keras.layers.BatchNormalization()(org_o)
    org_o = keras.layers.LeakyReLU(0.2)(org_o)

    rev_o = keras.layers.Conv2D(filters=filter_num * layer_dict[o.shape[1]], kernel_size=1, strides=1)(o)
    rev_o = keras.layers.BatchNormalization()(rev_o)
    rev_o = keras.layers.LeakyReLU(0.2)(rev_o)

    att = keras.layers.Activation("sigmoid")(-org_o)
    att_o = keras.layers.Multiply()([att, rev_o])
    att_o = keras.layers.Subtract()([org_o, att_o])

    rev_o = -rev_o
    for l, s in layer_dict.items():
        if l == img_shape[0]:
            break
        if l >= o.shape[1]:
            org_o = keras.layers.Conv2DTranspose(filters=filter_num * layer_dict[l * 2], kernel_size=3, strides=2, padding="SAME")(org_o)
            org_o = keras.layers.BatchNormalization()(org_o)
            org_o = keras.layers.LeakyReLU(0.2)(org_o)

            rev_o = keras.layers.Conv2DTranspose(filters=filter_num * layer_dict[l * 2], kernel_size=3, strides=2, padding="SAME")(rev_o)
            rev_o = keras.layers.BatchNormalization()(rev_o)
            rev_o = keras.layers.LeakyReLU(0.2)(rev_o)

            att_o = keras.layers.Conv2DTranspose(filters=filter_num * layer_dict[l * 2], kernel_size=3, strides=2, padding="SAME")(att_o)
            att_o = keras.layers.BatchNormalization()(att_o)
            att_o = keras.layers.LeakyReLU(0.2)(att_o)

    o = keras.layers.Concatenate(axis=-1)([org_o, rev_o, att_o])
    o = keras.layers.Conv2DTranspose(filters=256, kernel_size=1, strides=1, padding="SAME")(o)
    return keras.Model(inputs=inputs, outputs=o)


dataset = process_numpy(img_data, seg_data, batch_size=8)
model = get_model()
model.summary()
keras.utils.plot_model(model, show_shapes=True)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=CustomIoU(num_classes=256))
model.fit(dataset, epochs=20)
show(seg_data[:16], model(img_data[:16]))
