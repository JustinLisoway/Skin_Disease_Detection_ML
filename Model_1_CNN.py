from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# This is the 1st CNN model variation and was run on both the augmented and unaugmented data.

# Load data and preprocess
dataset_directory = '../../DermNet_Images/combined_v2'
input_shape = (224, 224, 3)
num_classes = 23
batch_size = 32
epochs = 20

data_generator = ImageDataGenerator(
    rescale=1. / 255,  # Rescale pixel values to [0, 1]
    rotation_range=15,  # Randomly rotate images by up to 15 degrees
    width_shift_range=0.1,  # Randomly shift image width by up to 10%
    height_shift_range=0.1,  # Randomly shift image height by up to 10%
    horizontal_flip=True,  # Randomly flip images horizontally
    validation_split=0.2  # 20% of the data will be used for validation
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

# Create Model with unique parameters
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Run model on whole dataset
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

print("Training accuracy:", history.history['accuracy'][-1])
print("Validation accuracy:", history.history['val_accuracy'][-1])


## Unaugmented data console output:
# Epoch 1/20
# 483/483 [==============================] - 326s 675ms/step - loss: 2.8711 - accuracy: 0.1499 - val_loss: 2.8859 - val_accuracy: 0.1445
# Epoch 2/20
# 483/483 [==============================] - 351s 726ms/step - loss: 2.7313 - accuracy: 0.1858 - val_loss: 2.8206 - val_accuracy: 0.1707
# Epoch 20/20
# 483/483 [==============================] - 431s 892ms/step - loss: 2.1048 - accuracy: 0.3620 - val_loss: 2.8505 - val_accuracy: 0.2060
# Training accuracy: 0.36204662919044495
# Validation accuracy: 0.2060275375843048

## Augmented data console output:
'''
Epoch 1/20
1011/1011 [==============================] - 714s 706ms/step - loss: 2.9511 - accuracy: 0.1237 - val_loss: 2.9691 - val_accuracy: 0.1354
Epoch 2/20
1011/1011 [==============================] - 725s 717ms/step - loss: 2.7591 - accuracy: 0.1843 - val_loss: 2.8323 - val_accuracy: 0.1718
Epoch 3/20
1011/1011 [==============================] - 757s 749ms/step - loss: 2.6391 - accuracy: 0.2190 - val_loss: 2.7542 - val_accuracy: 0.2083
Epoch 4/20
1011/1011 [==============================] - 949s 938ms/step - loss: 2.5496 - accuracy: 0.2450 - val_loss: 2.7430 - val_accuracy: 0.2153
Epoch 5/20
1011/1011 [==============================] - 899s 889ms/step - loss: 2.4751 - accuracy: 0.2654 - val_loss: 2.7395 - val_accuracy: 0.2217
Epoch 6/20
1011/1011 [==============================] - 880s 870ms/step - loss: 2.4114 - accuracy: 0.2831 - val_loss: 2.6690 - val_accuracy: 0.2393
Epoch 7/20
1011/1011 [==============================] - 915s 905ms/step - loss: 2.3518 - accuracy: 0.3011 - val_loss: 2.6591 - val_accuracy: 0.2391
Epoch 8/20
1011/1011 [==============================] - 805s 796ms/step - loss: 2.3017 - accuracy: 0.3142 - val_loss: 2.6244 - val_accuracy: 0.2502
Epoch 9/20
1011/1011 [==============================] - 734s 726ms/step - loss: 2.2460 - accuracy: 0.3368 - val_loss: 2.6187 - val_accuracy: 0.2728
Epoch 10/20
1011/1011 [==============================] - 762s 754ms/step - loss: 2.2084 - accuracy: 0.3455 - val_loss: 2.5962 - val_accuracy: 0.2693
Epoch 11/20
1011/1011 [==============================] - 764s 756ms/step - loss: 2.1555 - accuracy: 0.3594 - val_loss: 2.6000 - val_accuracy: 0.2731
Epoch 12/20
1011/1011 [==============================] - 732s 724ms/step - loss: 2.1146 - accuracy: 0.3744 - val_loss: 2.5656 - val_accuracy: 0.2747
Epoch 13/20
1011/1011 [==============================] - 762s 753ms/step - loss: 2.0696 - accuracy: 0.3863 - val_loss: 2.5280 - val_accuracy: 0.2892
Epoch 14/20
1011/1011 [==============================] - 803s 794ms/step - loss: 2.0435 - accuracy: 0.3921 - val_loss: 2.5492 - val_accuracy: 0.2761
Epoch 15/20
1011/1011 [==============================] - 749s 741ms/step - loss: 2.0052 - accuracy: 0.4047 - val_loss: 2.5248 - val_accuracy: 0.2934
Epoch 16/20
1011/1011 [==============================] - 776s 767ms/step - loss: 1.9768 - accuracy: 0.4127 - val_loss: 2.5024 - val_accuracy: 0.3031
Epoch 17/20
1011/1011 [==============================] - 823s 814ms/step - loss: 1.9518 - accuracy: 0.4244 - val_loss: 2.5129 - val_accuracy: 0.3122
Epoch 18/20
1011/1011 [==============================] - 834s 825ms/step - loss: 1.9182 - accuracy: 0.4280 - val_loss: 2.5561 - val_accuracy: 0.2875
Epoch 19/20
1011/1011 [==============================] - 1315s 1s/step - loss: 1.8831 - accuracy: 0.4368 - val_loss: 2.4760 - val_accuracy: 0.3169
Epoch 20/20
1011/1011 [==============================] - 1357s 1s/step - loss: 1.8640 - accuracy: 0.4473 - val_loss: 2.4849 - val_accuracy: 0.3175
Training accuracy: 0.4473375082015991
Validation accuracy: 0.3174780011177063
'''

