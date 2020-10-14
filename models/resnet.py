"""
    Src: https://github.com/suvoooo/Learn-TensorFlow/blob/master/resnet/Implement_Resnet_TensorFlow.ipynb
"""

from models.model_factory import Model
import tensorflow as tf
import numpy as np


class ResNet(Model):
    def __init__(self, num_blocks, num_classes, **kwargs):
        """
          num_blocks: [2, 2, 2, 2] : ResNet-18
                      [3, 4, 6, 3] : ResNet-50
                      [3, 4, 23, 3]: ResNet-101
                      [3, 8, 36, 3]: ResNet-152
        """

        super(ResNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_blocks = num_blocks

        if self.__class__ == ResNext:
          self.resblock = CardinalResBlock
        else:
          self.resblock = ResBlock
          self.cardinality = 0
          self.d = 0

        self.entry_block = [
                            tf.keras.layers.ZeroPadding2D((3,3)),
                            tf.keras.layers.Conv2D(64, (7,7), (2,2)),
                            tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.ReLU(),
                            tf.keras.layers.MaxPooling2D((3,3), (2,2))
                            ]
        
        self.blocks = self.prepare_blocks(self.num_blocks)
        self.avr = tf.keras.layers.AveragePooling2D((2,2), padding="SAME")
        self.fcn = FCN(self.num_classes)

    def prepare_blocks(self, num_blocks):
        filter = 64; d = self.d; blocks = []

        # since there is already maxpool in entry block, first stage will not downsample
        block = [ResConvBlock(strides=1, filters=(filter,filter*4))]
        for i in range(1, num_blocks[0]):
            block.append(self.resblock(filters=(filter, filter*4), c=self.cardinality, d=d))
        filter *= 2; d*= 2

        for n in range(1, len(num_blocks)):
          # first layer of the block downsamples
          block = [ResConvBlock(strides=2, filters=(filter,filter*4))]

          # rest of them keep the shape identical
          for i in range(1, num_blocks[n]):
            block.append(self.resblock(filters=(filter, filter*4), c=self.cardinality, d=d))
          blocks.append(block)

          filter *= 2; d *= 2
        return blocks

    def call(self, x):
        for layer in self.entry_block:
          x = layer(x)

        for block in self.blocks:
          for layer in block:
            x = layer(x)

        x = self.avr(x)
        return self.fcn(x)



class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResBlock, self).__init__()
        self.filters = filters

        self.blocks = [
                     # first block
                     [tf.keras.layers.Conv2D(self.filters[0], (1,1), (1,1), padding="VALID"),
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.ReLU()],
                     # second block
                     [tf.keras.layers.Conv2D(self.filters[0], (3,3), (1,1), padding="SAME"),
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.ReLU()],
                     # third block
                     [tf.keras.layers.Conv2D(self.filters[1], (1,1), (1,1), padding="VALID"),
                      tf.keras.layers.BatchNormalization()]
                     ]

        self.add = tf.keras.layers.Add()
        self.act = tf.keras.layers.ReLU()

    def call(self, inputs):
        skip = inputs

        for block in self.blocks:
            for layer in block:
                inputs = layer(inputs)

        output = self.add([inputs, skip])
        return self.act(output)



class ResConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides, **kwargs):
        super(ResConvBlock, self).__init__()
        self.filters = filters
        self.strides = strides

        self.blocks = [
                     # first block
                     [tf.keras.layers.Conv2D(self.filters[0], (1,1), (self.strides,self.strides), padding="VALID"),
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.ReLU()],
                     # second block
                     [tf.keras.layers.Conv2D(self.filters[0], (3,3), (1,1), padding="SAME"),
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.ReLU()],
                     # third block
                     [tf.keras.layers.Conv2D(self.filters[1], (1,1), (1,1), padding="VALID"),
                      tf.keras.layers.BatchNormalization()]
                     ]

        self.shortcut = [tf.keras.layers.Conv2D(self.filters[1], (1,1), (self.strides,self.strides), padding="VALID"),
                         tf.keras.layers.BatchNormalization()]
        self.add = tf.keras.layers.Add()
        self.act = tf.keras.layers.ReLU()

    def call(self, inputs):
      skip = inputs

      for block in self.blocks:
        for layer in block:
          inputs = layer(inputs)

      for layer in self.shortcut:
        skip = layer(skip)

      output = self.add([inputs, skip])
      return self.act(output)


"""
    ResNext architecture and cardinal residual block
"""
class ResNext(ResNet):

    def __init__(self, num_blocks, num_classes, cardinality, d, **kwargs):
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.cardinality = cardinality
        self.d = d
        super(ResNext, self).__init__(num_blocks, num_classes, **kwargs)


class CardinalResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, c, d, **kwargs):
        super(CardinalResBlock, self).__init__()
        self.filters = filters
        self.c = c
        self.d = d

        self.blocks = [
                     # first block
                     [tf.keras.layers.Conv2D(self.c * self.d, (1,1), (1,1), padding="VALID"),
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.ReLU()],
                     # second block
                     [[tf.keras.layers.Conv2D(self.d, (3,3), (1,1), padding="SAME") for i in range(self.c)],
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.ReLU()],
                     # third block
                     [tf.keras.layers.Conv2D(self.filters[1], (1,1), (1,1), padding="VALID"),
                      tf.keras.layers.BatchNormalization()]
                     ]

        self.conc = tf.keras.layers.Concatenate()
        self.add = tf.keras.layers.Add()
        self.act = tf.keras.layers.ReLU()

    def call(self, inputs):
        skip = inputs
        
        for i, block in enumerate(self.blocks):
            if i == 1:
                inputs = tf.split(inputs, self.c, axis=-1)
                inputs = self.conc([block[0][c](x) for c, x in enumerate(inputs)])
                inputs = block[1](inputs)
                inputs = block[2](inputs)
            else:
                for layer in block:
                    inputs = layer(inputs)

        output = self.add([inputs, skip])
        return self.act(output)
