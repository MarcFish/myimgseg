import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt


layer_dict = {4: 32, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}


def process_numpy(img_data, seg_data, batch_size=32):
    img_dataset = tf.data.Dataset.from_tensor_slices(img_data)
    seg_dataset = tf.data.Dataset.from_tensor_slices(seg_data)
    dataset = tf.data.Dataset.zip((img_dataset, seg_dataset)).shuffle(512).batch(batch_size=batch_size, drop_remainder=True).map(_process,tf.data.experimental.AUTOTUNE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def scale(image):
    return (image - 127.5) / 127.5


def rescale(image):
    return (image * 127.5 + 127.5).astype(np.uint8)


def _process(image, seg):
    img = tf.cast(image, tf.float32)
    img = scale(img)
    img = apply_augment(img)
    return img, seg


def apply_augment(image:tf.Tensor) -> tf.Tensor:
    image = tf.image.random_brightness(image, 0.5)
    image = tf.image.random_saturation(image, 0.0, 2.0)
    image = tf.image.random_contrast(image, 0.0, 1.0)
    # image = tf.image.random_flip_left_right(image)
    return image


def get_perceptual_func(model="vgg16"):
    if model == "vgg16":
        m = keras.applications.VGG16(include_top=False, pooling=None)
    elif model == "DenseNet201":
        m = keras.applications.DenseNet201(include_top=False, pooling=None)
    elif model == "EfficientNetB7":
        m = keras.applications.EfficientNetB7(include_top=False, pooling=None)
    elif model == "ResNet50":
        m = keras.applications.ResNet50(include_top=False, pooling=None)
    else:
        raise Exception("model not found")

    m.trainable = False

    def perceptual(pred, target):
        m_pred = m(pred)
        m_target = m(target)
        return tf.math.sqrt(tf.reduce_sum((m_pred-m_target)**2, axis=[1, 2, 3])) / tf.cast(tf.math.reduce_prod(m_pred.shape[1:]), tf.float32)

    return perceptual


class EMA:
    def __init__(self, model: keras.Model, tau=0.9):
        self.model = keras.models.clone_model(model)
        self.model.build(model.input_shape)
        self.tau = tau

    def register(self, model: keras.Model):
        for w, wt in zip(self.model.weights, model.weights):
            w.assign(wt)

    def update(self, model: keras.Model):
        for w, wt in zip(self.model.weights, model.weights):
            w.assign(self.tau * w + (1-self.tau) * wt)


def show(img, seg, n=16):
    plt.figure(figsize=(20, 4))
    seg = np.argmax(seg, axis=-1)[..., np.newaxis]
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(img[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(seg[i])
        plt.title("seg")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


class ShowCallBack(keras.callbacks.Callback):
    def __init__(self, img_sample, seg_sample):
        self.img_sample = img_sample
        self.seg_sample = seg_sample
        super(ShowCallBack, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        show(self.seg_sample, self.model(self.img_sample))


class CustomIoU(keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        super(CustomIoU, self).update_state(y_true, y_pred, sample_weight)
