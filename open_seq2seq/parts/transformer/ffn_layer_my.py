# Copyright 2018 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of fully connected network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FeedFowardNetwork(tf.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, relu_dropout, train, num_features, batch_size, regularizer=None):
    super(FeedFowardNetwork, self).__init__()
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout
    self.train = train
    self.num_features = num_features
    self.batch_size = batch_size

    # regularizer = tf.contrib.layers.l2_regularizer(0.0005)

    self.filter_dense_layer = tf.layers.Dense(
        filter_size,
        use_bias=True,
        activation=tf.nn.relu,
        name="filter_layer",
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer
    )
    self.output_dense_layer = tf.layers.Dense(
        hidden_size,
        use_bias=True,
        name="output_layer",
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer )

  def call(self, x, padding=None):
    # Retrieve dynamically known shapes
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    if padding is not None:
      with tf.name_scope("remove_padding"):
        # Flatten padding to [batch_size*length]
        pad_mask = tf.reshape(padding, [-1])

        nonpad_ids = tf.cast(tf.where(pad_mask < 1e-9), dtype=tf.int32)

        # Reshape x to [batch_size*length, hidden_size] to remove padding
        x = tf.reshape(x, [-1, self.hidden_size])
        x = tf.gather_nd(x, indices=nonpad_ids)

        # Reshape x from 2 dimensions to 3 dimensions.
        x.set_shape([None, self.hidden_size])
        x = tf.expand_dims(x, axis=0)

    output = self.filter_dense_layer(x)
    if self.train:
      output = tf.nn.dropout(output, keep_prob = 1 - self.relu_dropout)
    output = self.output_dense_layer(output)

    if padding is not None:
      with tf.name_scope("re_add_padding"):
        output = tf.squeeze(output, axis=0)
        output = tf.scatter_nd(
            indices=nonpad_ids,
            updates=output,
            shape=[batch_size * length * self.num_features, self.hidden_size]
        )
        if 'encoder' in x.name:
            output = tf.reshape(output, [self.batch_size, length, self.num_features, self.hidden_size])
        else:
            output = tf.reshape(output, [self.batch_size, length, self.hidden_size])
    return output

