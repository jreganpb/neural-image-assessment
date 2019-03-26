import os

from keras.models import Model
from keras.layers import Dense, Dropout
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras import backend as K

from utilities.keras_data_loader import train_image_paths, train_y, val_image_paths, val_y, image_generator

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

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

image_size = 224

base_model = keras.applications.Xception(input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = False

x = Dropout(0.75)(base_model.output)
x = Dense(10, activation='softmax')(x)

model = Model(base_model.input, x)
model.summary()
optimizer = Adam(lr=1e-3)
model.compile(optimizer, loss=earth_mover_loss)

# load weights from trained model if it exists
if os.path.exists('weights/xception_weights.h5'):
    model.load_weights('weights/xception_weights.h5')

# load pre-trained NIMA(Inception ResNet V2) classifier weights
# if os.path.exists('weights/inception_resnet_pretrained_weights.h5'):
#     model.load_weights('weights/inception_resnet_pretrained_weights.h5', by_name=True)

checkpoint = ModelCheckpoint('weights/xception_weights.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True,
                             mode='min')
tensorboard = TensorBoardBatch()
callbacks = [checkpoint, tensorboard]

batchsize = 256
epochs = 4

model.fit_generator(image_generator(files=train_image_paths,scores=train_y,batch_size=batchsize),
                    steps_per_epoch=len(train_image_paths) // batchsize, epochs=epochs,
                    validation_data=image_generator(files=val_image_paths,scores=val_y,batch_size=batchsize),
                    validation_steps=len(val_image_paths) // batchsize,callbacks=callbacks)