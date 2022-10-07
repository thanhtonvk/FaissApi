from keras import layers
from keras import models
from keras import regularizers
from keras import backend as K
import tensorflow as tf
def resnet_backbone(input_layer, conv_size_muls=[1, 2, 4, 4], start_neurons=16, dropout_rate=None):
    inner = None
    for index, i in enumerate(conv_size_muls):
        if index == 0:
            inner = input_layer
        inner = layers.Conv2D(start_neurons * i, (3, 3), activation=None, padding="same")(inner)
        inner = residual_block(inner, start_neurons * i)
        inner = residual_block(inner, start_neurons * i, True)
        inner = layers.MaxPooling2D((2, 2))(inner)
        if dropout_rate is not None:
            inner = layers.Dropout(dropout_rate)(inner)

    net = models.Model(inputs=[input_layer], outputs=inner)
    return net

def batch_activate(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = batch_activate(x)
    return x

def residual_block(block_input, num_filters=16, use_batch_activate=False):
    x = batch_activate(block_input)
    x = convolution_block(x, num_filters, (3, 3) )
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = layers.Add()([x, block_input])
    if use_batch_activate:
        x = batch_activate(x)
    return x
class ArcMarginProduct(layers.Layer) :
    def __init__(self, n_classes=1000, s=30.0, m=0.5, regularizer=None, **kwargs) :
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer= regularizers.get(regularizer)
        super(ArcMarginProduct, self).__init__(**kwargs)

    def build(self, input_shape) :
        self.W = self.add_weight(name='W',
                                shape=(input_shape[-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)
        super(ArcMarginProduct, self).build(input_shape)
        
    def call(self, input) :
        x = tf.nn.l2_normalize(input, axis=1)  
        W = tf.nn.l2_normalize(self.W, axis=1)
        logits = x @ W
        return K.clip(logits, -1+K.epsilon(), 1-K.epsilon())

    def compute_output_shape(self, input_shape) :
        return (None, self.n_classes)
def get_model():
    no_classes = 102
    s_value = 50.0
    m_value = 0.1
    emb_size = 128
    input_layer = layers.Input(name='the_input', shape=(384, 384, 3), dtype='float32')
    base_net = resnet_backbone(
        input_layer, conv_size_muls=[1, 1, 2, 2, 4, 4],
        start_neurons=32, dropout_rate=None
    )

    inner = layers.GlobalAveragePooling2D()(base_net.output)
    inner = layers.Dropout(rate=0.25)(inner)
    inner = layers.Dense(emb_size, name='embedding')(inner)
    inner = layers.BatchNormalization()(inner)
    inner = layers.Dropout(0.25)(inner)
    output = ArcMarginProduct(102, s=s_value, m=m_value)(inner)
    model = models.Model(inputs=base_net.input, outputs=output)
    pred_model = models.Model(inputs=[model.input], outputs=model.layers[-3].output)
    pred_model.load_weights("./emb_arc.h5")
    return pred_model