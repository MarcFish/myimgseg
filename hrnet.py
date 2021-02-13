import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import numpy as np

from utils import layer_dict, process_numpy, show, ShowCallBack, CustomIoU


img_data = np.load("./seg_dataset/img128.npy")
seg_data = np.load("./seg_dataset/seg128.npy")


def get_model(img_shape=(128, 128, 3), filter_num=16, version="v1"):
    inputs = keras.layers.Input(shape=(img_shape))
    o = inputs
    o_dict = dict()
    l_list = []
    for l, s in reversed(layer_dict.items()):
        if l == img_shape[0]:
            o = keras.layers.Conv2D(filters=filter_num * s, kernel_size=3, strides=1, padding="SAME")(o)
            o = keras.layers.BatchNormalization()(o)
            o = keras.layers.LeakyReLU(0.2)(o)
            o_dict[l] = o
            l_list.append(l)

        if l < img_shape[0]:
            l_list.append(l)
            t_dict = dict()
            for l in l_list[:-1]:
                for l_ in l_list:
                    t_dict.setdefault(l_, [])
                    if l == l_:
                        o = keras.layers.Conv2D(filters=filter_num * layer_dict[l_], kernel_size=3, strides=1,
                                                padding="SAME")(o_dict[l])
                        o = keras.layers.BatchNormalization()(o)
                        o = keras.layers.LeakyReLU(0.2)(o)
                        t_dict[l_].append(o)
                    elif l > l_:
                        o = keras.layers.Conv2D(filters=filter_num * layer_dict[l_], kernel_size=3, strides=l // l_,
                                                 padding="SAME")(o_dict[l])
                        o = keras.layers.BatchNormalization()(o)
                        o = keras.layers.LeakyReLU(0.2)(o)
                        t_dict[l_].append(o)
                    else:
                        o = keras.layers.Conv2DTranspose(filters=filter_num * layer_dict[l_], kernel_size=3,
                                                          strides=l_ // l, padding="SAME")(o_dict[l])
                        o = keras.layers.BatchNormalization()(o)
                        o = keras.layers.LeakyReLU(0.2)(o)
                        t_dict[l_].append(o)

            for l, os in t_dict.items():
                if len(os) == 1:
                    o = os[0]
                else:
                    o = keras.layers.Concatenate(axis=-1)(os)

                o = keras.layers.Conv2D(filters=filter_num * layer_dict[l], kernel_size=3, strides=1, padding="SAME")(o)
                o = keras.layers.BatchNormalization()(o)
                o = keras.layers.LeakyReLU(0.2)(o)

                o_dict[l] = o
    if version == "v1":
        o = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="SAME")(o_dict[img_shape[0]])
    elif version == "v2":
        os = []
        for l, o in o_dict.items():
            o_ = keras.layers.Conv2DTranspose(filters=filter_num * layer_dict[l], kernel_size=3, strides=img_shape[0]//l, padding="SAME")(o)
            o_ = keras.layers.BatchNormalization()(o_)
            o_ = keras.layers.LeakyReLU(0.2)(o_)
            os.append(o_)
        o = keras.layers.Concatenate(axis=-1)(os)
        o = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="SAME")(o)
    else:
        raise Exception("wrong version")
    return keras.Model(inputs=inputs, outputs=o)


dataset = process_numpy(img_data, seg_data, batch_size=8)
model = get_model()
model.summary()
keras.utils.plot_model(model, show_shapes=True)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=CustomIoU(num_classes=256))
model.fit(dataset, epochs=200)
show(seg_data[:16], model(img_data[:16]))
