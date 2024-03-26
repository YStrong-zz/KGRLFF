# -*- coding: utf-8 -*-

from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf

class SumAggregator1(Layer):
    def __init__(self, activation: str ='relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(SumAggregator1, self).__init__(**kwargs)
        self.activation_name = activation
        if activation == 'relu':
            self.activation = K.relu
        elif activation == 'tanh':
            self.activation = K.tanh
        else:
            raise ValueError(f'`activation` not understood: {activation}')
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        ent_embed_dim = input_shape[0][-1]
     
        self.w = self.add_weight(name=self.name+'_w', shape=(ent_embed_dim, ent_embed_dim),
                                 initializer=self.initializer, regularizer=self.regularizer)
        self.b = self.add_weight(name=self.name+'_b', shape=(ent_embed_dim,), initializer='zeros')
        super(SumAggregator1, self).build(input_shape)

    def call(self, inputs, **kwargs):
        entity, neighbor, path = inputs
        # return self.activation(K.dot((entity + neighbor + path), self.w) + self.b)
        if self.activation_name == 'tanh':
            return self.activation(K.dot((entity + neighbor + path), self.w) + self.b)
        else:
            return self.activation(K.dot((entity + neighbor), self.w) + self.b)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class ConcatAggregator1(Layer):
    def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(ConcatAggregator1, self).__init__(**kwargs)
        self.activation_name = activation
        if activation == 'relu':
            self.activation = K.relu
        elif activation == 'tanh':
            self.activation = K.tanh
        else:
            raise ValueError(f'`activation` not understood: {activation}')
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        ent_embed_dim = input_shape[0][-1]
        neighbor_embed_dim = input_shape[1][-1]
        path_embed_dim = input_shape[2][-1]
    
        self.w = self.add_weight(name=self.name + '_w',
                                 shape=(ent_embed_dim+neighbor_embed_dim+path_embed_dim, ent_embed_dim),
                                 initializer=self.initializer, regularizer=self.regularizer)
        self.v = self.add_weight(name=self.name + '_v',
                                 shape=(ent_embed_dim+neighbor_embed_dim, ent_embed_dim),
                                 initializer=self.initializer, regularizer=self.regularizer)
        self.b = self.add_weight(name=self.name + '_b', shape=(ent_embed_dim,),
                                 initializer='zeros')
        super(ConcatAggregator1, self).build(input_shape)

    def call(self, inputs, **kwargs):
        entity, neighbor, path = inputs
    
        # return self.activation(K.dot(K.concatenate([entity, neighbor, path]), self.w) + self.b)

        if self.activation_name == 'tanh':
            return self.activation(K.dot(K.concatenate([entity, neighbor, path]), self.w) + self.b)
        else:
            return self.activation(K.dot(K.concatenate([entity, neighbor]), self.v) + self.b)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class NeighAggregator1(Layer):
    def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(NeighAggregator1, self).__init__()
        if activation == 'relu':
            self.activation = K.relu
        elif activation == 'tanh':
            self.activation = K.tanh
        else:
            raise ValueError(f'`activation` not understood: {activation}')
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        ent_embed_dim = input_shape[0][-1]
        neighbor_embed_dim = input_shape[1][-1]
        self.w = self.add_weight(name=self.name + '_w',
                                 shape=(neighbor_embed_dim, ent_embed_dim),
                                 initializer=self.initializer, regularizer=self.regularizer)
        self.b = self.add_weight(name=self.name + '_b', shape=(ent_embed_dim,),
                                 initializer='zeros')
        super(NeighAggregator1, self).build(input_shape)

    def call(self, inputs, **kwargs):
        entity, neighbor, path = inputs
        return self.activation(K.dot(neighbor, self.w) + self.b)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

