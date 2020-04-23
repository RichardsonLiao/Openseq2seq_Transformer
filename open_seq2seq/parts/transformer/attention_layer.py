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
"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Attention(tf.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(
          self,
          hidden_size,
          num_heads,
          attention_dropout,
          train,
          batch_size,
          num_feature,
          mode="loung",
          regularizer=None,
          window_size=None,
          back_step_size=None
  ):
    if hidden_size % num_heads != 0:
      raise ValueError("Hidden size must be evenly divisible by the number of "
                       "heads.")

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.train = train
    self.mode = mode
    self.batch_size = batch_size
    self.num_feature = num_feature

    # Parameters for monotonic attention forcing during inference
    self.window_size = window_size
    self.back_step_size = back_step_size

    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q",
                                         kernel_regularizer=regularizer)
    self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k",
                                         kernel_regularizer=regularizer)
    self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v",
                                         kernel_regularizer=regularizer)
    self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                              name="output_transform",
                                              kernel_regularizer=regularizer)

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, num_feature, hidden_size]
    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads, num_feature]
    """
    CSI="\x1B["
    print(CSI+"32;40m" + "open_seq2seq/parts/transformer/attention_layer.py line 74" + CSI + "0m")
    print('x')
    print(x)
    with tf.name_scope("split_heads"):
      #batch_size = tf.shape(x)[0]
      batch_size = self.batch_size
      length = tf.shape(x)[1]
      num_feature = self.num_feature

      # Calculate depth of last dimension after it has been split.
      depth = (self.hidden_size // self.num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth, num_feature])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3, 4])

  def combine_heads(self, x):
    """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads, num_feature]
    Returns:
      A tensor with shape [batch_size, length, hidden_size, num_feature]
    """
    CSI="\x1B["
    print(CSI+"32;40m" + "open_seq2seq/parts/transformer/attention_layer.py line 99" + CSI + "0m")
    print('x')
    print(x)
    with tf.name_scope("combine_heads"):
      #batch_size = tf.shape(x)[0]
      batch_size = self.batch_size
      length = tf.shape(x)[2]
      num_feature = self.num_feature
      x = tf.transpose(x, [0, 2, 4, 1, 3])  # --> [batch, length, num_heads, depth, num_feature]
      return tf.reshape(x, [batch_size, length, num_feature, self.hidden_size])

  def call(self, x, y, bias, cache=None, positions=None):
    """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
      positions: decoder-encoder alignment for previous steps [batch_size, n_heads, length_x]
    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1)
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    if self.mode == "loung":
      # Scale q to prevent the dot product between q and k from growing too large.
      depth = (self.hidden_size // self.num_heads)
      q *= 256 ** -0.5

      # Calculate dot product attention
      # logits = tf.matmul(q, k, transpose_b=True)
      # logits += bias
      # weights = tf.nn.softmax(logits, name="attention_weights")
      CSI="\x1B["
      print(CSI+"32;40m" + "open_seq2seq/parts/transformer/attention_layer.py line 182" + CSI + "0m")
      print('q')
      print(q)
      CSI="\x1B["
      print(CSI+"32;40m" + "open_seq2seq/parts/transformer/attention_layer.py line 186" + CSI + "0m")
      print('k')
      print(k)
      #logits = tf.matmul(q, k)
      #logits = tf.matmul(q, k, transpose_b=True)
      logits = tf.multiply(q, k)
      CSI="\x1B["
      print(CSI+"32;40m" + "open_seq2seq/parts/transformer/attention_layer.py line 193" + CSI + "0m")
      print('logits')
      print(logits)
      dtype = logits.dtype
      if dtype != tf.float32:
        CSI="\x1B["
        print(CSI+"32;40m" + "open_seq2seq/parts/transformer/attention_layer.py line 165" + CSI + "0m")
        print('bias')
        print(bias)
        # upcast softmax inputs
        logits = tf.cast(x=logits, dtype=tf.float32)
        logits += bias
        logits = logits/16 # divided by sruare root of d(att)
        weights = tf.nn.softmax(logits, name="attention_weights")
        # downcast softmax output
        weights = tf.cast(weights, dtype=dtype)
        CSI="\x1B["
        print(CSI+"32;40m" + "open_seq2seq/parts/transformer/attention_layer.py line 171" + CSI + "0m")
        print('weights')
        print(weights)
      else:
        # Logits shape: [batch, head, decoder, encoder]
        # Bias shape:   [batch, 1, 1, encoder]

        # Force monotonic attention during inference
        if positions is not None and self.window_size is not None:
          assert self.back_step_size is not None

          max_length = tf.shape(logits)[-1]

          # Allow to make back_step_size steps back
          window_pos = tf.maximum(positions - self.back_step_size, tf.zeros_like(positions))

          # Create attention mask
          mask_large = tf.sequence_mask(window_pos + self.window_size, maxlen=max_length)
          mask_large = tf.cast(mask_large, tf.float32)
          mask_small = tf.sequence_mask(window_pos, maxlen=max_length)
          mask_small = tf.cast(mask_small, tf.float32)
          mask = mask_large - mask_small
          mask = -1e9 * (1 - mask)

          bias = mask + bias

          # Clipping
          bias = tf.maximum(bias, -1e9)

        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
    elif self.mode == "bahdanau":
      att_v = tf.get_variable(
        "attention_v", [self.hidden_size // self.num_heads], dtype=q.dtype
      )

      # Compute the attention score
      if bias is not None:
        weights = tf.reduce_sum(
          tf.nn.tanh(att_v * tf.nn.tanh(k + q + bias)), 3
        )
      else:
        weights = tf.reduce_sum(
          tf.nn.tanh(att_v * tf.nn.tanh(k + q)), 3
        )
      weights = tf.nn.softmax(weights)
      weights = tf.expand_dims(weights, 2)
    else:
      raise ValueError(
        "Mode for multi-head attention must be either loung for dot-product",
        "attention, or bahdanau for content-based/additive/mlp-base attention"
      )

    if self.train:
      weights = tf.nn.dropout(weights, keep_prob=1 - self.attention_dropout)
    CSI="\x1B["
    print(CSI+"32;40m" + "open_seq2seq/parts/transformer/attention_layer.py line 227" + CSI + "0m")
    print('weights')
    print(weights)
    CSI="\x1B["
    print(CSI+"32;40m" + "open_seq2seq/parts/transformer/attention_layer.py line 211" + CSI + "0m")
    print('v')
    print(v)
    #attention_output = tf.matmul(weights, v)
    #attention_output = tf.matmul(weights, v, transpose_b=True)
    attention_output = tf.multiply(weights, v)

    CSI="\x1B["
    print(CSI+"32;40m" + "open_seq2seq/parts/transformer/attention_layer.py line 253" + CSI + "0m")
    print('attention_output')
    print(attention_output)
    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = self.combine_heads(attention_output)
    CSI="\x1B["
    print(CSI+"32;40m" + "open_seq2seq/parts/transformer/attention_layer.py line 259" + CSI + "0m")
    print('attention_output')
    print(attention_output)

    # Run the combined outputs through another linear projection layer.
    attention_output = self.output_dense_layer(attention_output)
    CSI="\x1B["
    print(CSI+"32;40m" + "open_seq2seq/parts/transformer/attention_layer.py line 266" + CSI + "0m")
    print('attention_output')
    print(attention_output)
    return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, x, bias, cache=None):
    return super(SelfAttention, self).call(x, x, bias, cache)
