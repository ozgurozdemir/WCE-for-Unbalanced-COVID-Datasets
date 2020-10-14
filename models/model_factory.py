import numpy as np
import time
import os

import tensorflow as tf


class Model(tf.keras.Model):
  def __init__(self, optimizer, wce_beta=None, checkpoint_dir=None, load_checkpoint=False):
    super(Model, self).__init__()
    self.optimizer = optimizer
    self.wce_beta = wce_beta
    self.checkpoint_dir = checkpoint_dir
    self.load_checkpoint = load_checkpoint

    if checkpoint_dir: self.create_checkpoint()
    if self.wce_beta: self.loss_function = self.weighted_cross_entropy 
    else: self.loss_function =  tf.keras.losses.CategoricalCrossentropy(from_logits=True)


  def create_checkpoint(self):
    checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,model=self)
    self.ckpt_manager = tf.train.CheckpointManager(checkpoint, self.checkpoint_dir, max_to_keep=1)

    if self.load_checkpoint and self.ckpt_manager.latest_checkpoint:
      checkpoint.restore(self.ckpt_manager.latest_checkpoint)
      print ('>> Latest checkpoint restored!!')


  def calculate_loss(self, output, target):
    output = tf.cast(output, dtype=tf.float32)
    target = tf.cast(target, dtype=tf.float32)
    return self.loss_function(output, target)

  
  def weighted_cross_entropy(self, output, target):
    #  to ensure stability and avoid overflow
    output = tf.clip_by_value(output, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    output = tf.math.log(output / (1 - output))

    # one-hot encoding to label (sparse to non-sparse)
    output *= [0, 1]
    target *= [0, 1]

    # weighted ce: labels * -log(sigmoid(logits)) * BETA + (1-labels) * -log(1-sigmoid(logits))
    loss = target * -tf.math.log(tf.math.sigmoid(output)) * self.wce_beta + (1-target) * -tf.math.log(1-tf.math.sigmoid(output))
    loss = tf.reduce_sum(loss, axis=0)
    loss = tf.reduce_mean(loss)
    return loss


  @tf.function
  def train_step(self, input_image, target, train=True):
    with tf.GradientTape() as tape:
      output = self(input_image, training=train)
      loss = self.calculate_loss(output, target)

    if train:
      gradients = tape.gradient(loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return loss


  def fit(self, train_ds, test_ds, epochs, save=False):
    """
        returns: best model
    """
    best_valid_loss = 1000

    for epoch in range(epochs):
      start = time.time()
      loss, val_loss = [], []  

      # Training
      print("Epoch: ", epoch + 1)
      for n, (input_image, target) in train_ds.enumerate():
        batch_loss = self.train_step(input_image, target, train=True)
        loss.append(batch_loss)  

      # Validation
      for n, (input_image, target) in test_ds.enumerate():
        batch_loss = self.train_step(input_image, target, train=False)
        val_loss.append(batch_loss)  

      if np.mean(val_loss) < best_valid_loss:
        best_valid_loss = np.mean(val_loss)
        best_model = self
        if save:
          print(f">> Checkpoint is saved in epoch {epoch+1}, val loss: {best_valid_loss} ")
          self.ckpt_manager.save()

      # Time
      print(f'Epoch {epoch + 1} => Loss {np.mean(loss):.4f} -- Val Loss {np.mean(val_loss):.4f}')
      print(f'Time taken for epoch {epoch + 1} is {time.time()-start} sec..., Best val loss {best_valid_loss}\n')

    return best_model


"""
  Fully Connected Layer for classification. Generates sparse matrix of classes
"""
class FCN (tf.keras.layers.Layer):
  def __init__(self, num_classes=2):
    super(FCN, self).__init__()
    self.num_classes = num_classes

    self.fcn = []
    self.fcn.append(tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid'))
    self.fcn.append(tf.keras.layers.Flatten())
    self.fcn.append(tf.keras.layers.Dense(128))
    self.fcn.append(tf.keras.layers.Dropout(0.2))
    self.fcn.append(tf.keras.layers.BatchNormalization())
    self.fcn.append(tf.keras.layers.Activation('relu'))
    self.fcn.append(tf.keras.layers.Dense(16))
    self.fcn.append(tf.keras.layers.Dropout(0.2))
    self.fcn.append(tf.keras.layers.BatchNormalization())
    self.fcn.append(tf.keras.layers.Activation('relu'))
    self.fcn.append(tf.keras.layers.Dense(self.num_classes, activation="softmax"))

  def call(self, inputs):
    for layer in self.fcn:
      inputs = layer(inputs)
    return inputs


"""
  Preassembled CNN architectures, e.g. ResNet, VGG, Xception etc.
"""
class PreassembledCNN(Model):
  def __init__(self, arch, num_classes=2, **kwargs):
    super(PreassembledCNN, self).__init__(**kwargs)
    self.num_classes = num_classes
    self.arch = arch
    self.fcn = FCN(self.num_classes)

  def call(self, inputs):
    outputs = self.arch(inputs)
    return self.fcn(outputs)