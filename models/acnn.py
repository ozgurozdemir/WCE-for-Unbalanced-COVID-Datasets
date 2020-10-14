"""
  Atrous CNN Architecture. Source:https://arxiv.org/pdf/1901.09203.pdf
"""

from models.model_factory import Model, FCN
import tensorflow as tf
import numpy as np


class ACNN(Model):
  # Atrous Convolutional Layer
  class Atrous_Conv2d (tf.keras.layers.Layer):
    def __init__(self, filters, rate):
      super(ACNN.Atrous_Conv2d, self).__init__()
      self.filters = tf.Variable(tf.random.truncated_normal(filters, stddev=0.5), trainable=True)
      self.rate = rate

    def call(self, x):
      return tf.nn.atrous_conv2d(x, self.filters, rate=self.rate, padding="SAME")

  # Block of Atrous Convolutional Layers
  class Atrous_Block (tf.keras.layers.Layer):
    def __init__(self):
      super(ACNN.Atrous_Block, self).__init__()
      self.a1 = ACNN.Atrous_Conv2d(filters=(3, 3, 3, 9), rate=1)
      self.n1 = tf.keras.layers.BatchNormalization()
      self.r1 = tf.keras.layers.ReLU()

      self.a2 = ACNN.Atrous_Conv2d(filters=(3, 3, 9, 3), rate=3)
      self.n2 = tf.keras.layers.BatchNormalization()
      self.r2 = tf.keras.layers.ReLU()
      self.add = tf.keras.layers.Add()

    def call(self, x):
      out = self.a1(x)
      out = self.n1(out)
      out = self.r1(out)
      out = self.a2(out)
      out = self.n2(out)
      out = self.add([x, out])
      return out

  def __init__(self, block_size, num_classes, **kwargs):
    super(ACNN, self).__init__(**kwargs)
    self.block_size = block_size
    self.num_classes = num_classes

    self.blocks = []
    for i in range(self.block_size):
      self.blocks.append(ACNN.Atrous_Block())
    self.fcn = FCN(self.num_classes)

  def call(self, inputs):
    for i in range(self.block_size):
      inputs = self.blocks[i](inputs)
    return self.fcn(inputs)
