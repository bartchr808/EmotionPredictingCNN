# Based off of implementation of PReLU in keras' github repo

import numpy as np
from keras.layers import initializers
from keras.layers import regularizers
from keras.layers import constraints
from keras.engine import InputSpec
from keras.engine.topology import Layer
import tensorflow as tf

class APLUnit(Layer):
    """Adaptive Piecewise Linear Units.
    It follows:
    `f(x) = max(0,x) + \sum\limits_{s=1}^S (a_i^s * max(0, -x + b_i^s))`,
    where `S` is a tuned hyperparameter and `s` is an index (not a exponent) for each neuron `i`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        S: number of piecewise functions.
        alpha and b _initializer: initializer function for the weights.
        alpha and b _regularizer: regularizer for the weights.
        alpha and b _constraint: constraint for the weights.
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
    # References
        - [Keras source code for PReLU](https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py#L49)
        - [Learning Activation Functions to Improve Deep Neural Networks](https://arxiv.org/pdf/1412.6830.pdf)
    """

    def __init__(self, alpha_initializer='zeros',
                 b_initializer='zeros',
                 S=1,
                 alpha_regularizer=None,
                 b_regularizer=None,
                 alpha_constraint=None,
                 b_constraint=None,
                 shared_axes=None,
                 **kwargs):
        super(APLUnit, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.b_initializer = initializers.get(b_initializer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.b_constraint = constraints.get(b_constraint)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        self.S = S
        self.alpha_arr=[]
        self.b_arr=[]

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True
        for i in range(self.S):
            self.alpha_arr.append(self.add_weight(shape=param_shape,
                                        name='alpha_' + str(i),
                                        initializer=self.alpha_initializer,
                                        regularizer=self.alpha_regularizer,
                                        constraint=self.alpha_constraint))
            self.b_arr.append(self.add_weight(shape=param_shape,
                                        name='b_' + str(i),
                                        initializer=self.b_initializer,
                                        regularizer=self.b_regularizer,
                                        constraint=self.b_constraint))
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs, mask=None):
        max_a = tf.maximum(0., inputs)
        max_b = 0
        for i in range(self.S):
            max_b += self.alpha_arr[i] * tf.maximum(0., -inputs + self.b_arr[i])

        return max_a + max_b

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.b_initializer),
            'alpha_regularizer': regularizers.serialize(self.b_regularizer),
            'alpha_constraint': constraints.serialize(self.b_constraint),
            'b_initializer': initializers.serialize(self.b_initializer),
            'b_regularizer': regularizers.serialize(self.b_regularizer),
            'b_constraint': constraints.serialize(self.b_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(APLUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
