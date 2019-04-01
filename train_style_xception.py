import os

from keras.models import Model
from keras.layers import Dense, Dropout
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from utilities.keras_style_loader import train_labels, train_files, test_labels, test_files, image_generator

'''
Below is a modification to the TensorBoard callback to perform
batchwise writing to the tensorboard, instead of only at the end
of the batch.
'''
class TensorBoardBatch(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(TensorBoardBatch, self).__init__(*args)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, batch)

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch * self.batch_size)

        self.writer.flush()

## Dataset is much smaller here, so we can load dataset in its entirety

image_size = 224

base_model = keras.applications.Xception(input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = False

x = Dropout(0.75)(base_model.output)
x = Dense(14, activation='sigmoid')(x)

model = Model(base_model.input, x)
#model.summary()
optimizer = Adam(lr=1e-3)
model.compile(optimizer, loss='binary_crossentropy')

# load weights from trained model if it exists
if os.path.exists('style_weights/xception_weights.h5'):
    model.load_weights('style_weights/xception_weights.h5')

# load pre-trained NIMA(Inception ResNet V2) classifier weights
# if os.path.exists('weights/inception_resnet_pretrained_weights.h5'):
#     model.load_weights('weights/inception_resnet_pretrained_weights.h5', by_name=True)

checkpoint = ModelCheckpoint('style_weights/xception_weights.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True,
                             mode='min')
tensorboard = TensorBoardBatch()
callbacks = [checkpoint, tensorboard]

batchsize = 256
epochs = 20

model.fit_generator(image_generator(files=train_files,scores=train_labels,batch_size=batchsize),
                    steps_per_epoch=len(train_files) // batchsize, epochs=epochs,
                    validation_data=image_generator(files=test_files,scores=test_labels,batch_size=batchsize),
                    validation_steps=len(test_files) // batchsize,callbacks=callbacks)