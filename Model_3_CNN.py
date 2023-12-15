from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# This is the 3rd CNN model variation and was only run on the augmented data.

# Load data and preprocess
dataset_directory = '../../DermNet_Images/combined_v2'
input_shape = (224, 224, 3)
num_classes = 23
batch_size = 32
epochs = 20

data_generator = ImageDataGenerator(
    rescale=1./255,      # Rescale pixel values to [0, 1]
    rotation_range=15,   # Randomly rotate images by up to 15 degrees
    width_shift_range=0.1,  # Randomly shift image width by up to 10%
    height_shift_range=0.1,  # Randomly shift image height by up to 10%
    horizontal_flip=True,    # Randomly flip images horizontally
    validation_split=0.2    # 20% of the data will be used for validation
)

train_generator = data_generator.flow_from_directory(
    dataset_directory,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = data_generator.flow_from_directory(
    dataset_directory,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Create model with altered parameters
model = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Run the model on whole dataset
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, batch_size=64)

print("Training accuracy:", history.history['accuracy'][-1])
print("Validation accuracy:", history.history['val_accuracy'][-1])

## Augmented data console output:
"""
Epoch 1/20
483/483 [==============================] - 3089s 6s/step - loss: 2.8976 - accuracy: 0.1342 - val_loss: 2.8700 - val_accuracy: 0.1538
Epoch 2/20
483/483 [==============================] - 2969s 6s/step - loss: 2.7633 - accuracy: 0.1762 - val_loss: 2.8187 - val_accuracy: 0.1637
Epoch 3/20
483/483 [==============================] - 2917s 6s/step - loss: 2.6732 - accuracy: 0.2076 - val_loss: 2.7534 - val_accuracy: 0.1808
Epoch 4/20
483/483 [==============================] - 2989s 6s/step - loss: 2.5992 - accuracy: 0.2277 - val_loss: 2.7249 - val_accuracy: 0.1897
Epoch 5/20
483/483 [==============================] - 3029s 6s/step - loss: 2.5406 - accuracy: 0.2448 - val_loss: 2.7276 - val_accuracy: 0.1998
Epoch 6/20
483/483 [==============================] - 1962s 4s/step - loss: 2.4932 - accuracy: 0.2599 - val_loss: 2.7124 - val_accuracy: 0.2050
Epoch 7/20
483/483 [==============================] - 1364s 3s/step - loss: 2.4495 - accuracy: 0.2755 - val_loss: 2.6926 - val_accuracy: 0.1982
Epoch 8/20
483/483 [==============================] - 1382s 3s/step - loss: 2.4069 - accuracy: 0.2808 - val_loss: 2.6719 - val_accuracy: 0.2099
Epoch 9/20
483/483 [==============================] - 1365s 3s/step - loss: 2.3659 - accuracy: 0.2934 - val_loss: 2.6700 - val_accuracy: 0.2203
Epoch 10/20
483/483 [==============================] - 1371s 3s/step - loss: 2.3331 - accuracy: 0.3040 - val_loss: 2.6629 - val_accuracy: 0.2182
Epoch 11/20
483/483 [==============================] - 1383s 3s/step - loss: 2.2906 - accuracy: 0.3154 - val_loss: 2.6683 - val_accuracy: 0.2203
Epoch 12/20
483/483 [==============================] - 1358s 3s/step - loss: 2.2618 - accuracy: 0.3227 - val_loss: 2.6788 - val_accuracy: 0.2154
Epoch 13/20
483/483 [==============================] - 1357s 3s/step - loss: 2.2152 - accuracy: 0.3311 - val_loss: 2.6592 - val_accuracy: 0.2297
Epoch 14/20
483/483 [==============================] - 1331s 3s/step - loss: 2.1762 - accuracy: 0.3413 - val_loss: 2.6519 - val_accuracy: 0.2245
Epoch 15/20
483/483 [==============================] - 1333s 3s/step - loss: 2.1492 - accuracy: 0.3512 - val_loss: 2.7021 - val_accuracy: 0.2164
Epoch 16/20
483/483 [==============================] - 1355s 3s/step - loss: 2.1031 - accuracy: 0.3650 - val_loss: 2.6967 - val_accuracy: 0.2356
Epoch 17/20
483/483 [==============================] - 1334s 3s/step - loss: 2.0694 - accuracy: 0.3765 - val_loss: 2.7025 - val_accuracy: 0.2266
Epoch 18/20
483/483 [==============================] - 1336s 3s/step - loss: 2.0263 - accuracy: 0.3852 - val_loss: 2.7077 - val_accuracy: 0.2369
Epoch 19/20
483/483 [==============================] - 1329s 3s/step - loss: 1.9877 - accuracy: 0.3948 - val_loss: 2.6863 - val_accuracy: 0.2349
Epoch 20/20
483/483 [==============================] - 1325s 3s/step - loss: 1.9601 - accuracy: 0.4045 - val_loss: 2.7119 - val_accuracy: 0.2349
Training accuracy: 0.40446892380714417
Validation accuracy: 0.234866201877594
"""
