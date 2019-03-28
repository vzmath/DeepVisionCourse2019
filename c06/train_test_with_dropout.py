from build_data_dir import train_dir, validation_dir
from models import convnet_with_dropout

from keras.preprocessing.image import ImageDataGenerator

# parameters for data augmentation
RESCALE = 1./255
ROTATION_RANGE = 40
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 20

# model train parameters
DROPOUT_RATE = 0.5
NUM_EPOCHS = 100

# data preprocessing
# data augmentation generator
train_datagen = ImageDataGenerator(
        rescale=RESCALE,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        shear_range=SHEAR_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=HORIZONTAL_FLIP)
test_datagen = ImageDataGenerator(rescale=RESCALE)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary')

model = convnet_with_dropout(dropout_rate=DROPOUT_RATE)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=50)
model.save('saved_models/cats_and_dogs_small_2.h5')
