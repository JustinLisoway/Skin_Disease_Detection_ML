from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# This is the 2nd CNN model variation and was run on both the augmented and unaugmented data.

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
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # Add dropout for regularization
    layers.Dense(num_classes, activation='softmax')
])

# Also try compiling the model with a lower learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Run the model on whole dataset
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, batch_size=64)

print("Training accuracy:", history.history['accuracy'][-1])
print("Validation accuracy:", history.history['val_accuracy'][-1])


## Unaugmented data console output:
"""
Epoch 1/20
483/483 [==============================] - 823s 2s/step - loss: 2.9316 - accuracy: 0.1291 - val_loss: 2.8641 - val_accuracy: 0.1442
Epoch 2/20
483/483 [==============================] - 841s 2s/step - loss: 2.8197 - accuracy: 0.1663 - val_loss: 2.8334 - val_accuracy: 0.1611
Epoch 3/20
483/483 [==============================] - 825s 2s/step - loss: 2.7489 - accuracy: 0.1845 - val_loss: 2.7969 - val_accuracy: 0.1673
Epoch 4/20
483/483 [==============================] - 776s 2s/step - loss: 2.7029 - accuracy: 0.2001 - val_loss: 2.7735 - val_accuracy: 0.1842
Epoch 5/20
483/483 [==============================] - 812s 2s/step - loss: 2.6665 - accuracy: 0.2147 - val_loss: 2.7395 - val_accuracy: 0.1907
Epoch 6/20
483/483 [==============================] - 808s 2s/step - loss: 2.6322 - accuracy: 0.2192 - val_loss: 2.7606 - val_accuracy: 0.1813
Epoch 7/20
483/483 [==============================] - 801s 2s/step - loss: 2.6060 - accuracy: 0.2322 - val_loss: 2.7272 - val_accuracy: 0.1923
Epoch 8/20
483/483 [==============================] - 777s 2s/step - loss: 2.5783 - accuracy: 0.2387 - val_loss: 2.7390 - val_accuracy: 0.1876
Epoch 9/20
483/483 [==============================] - 789s 2s/step - loss: 2.5566 - accuracy: 0.2399 - val_loss: 2.7102 - val_accuracy: 0.1964
Epoch 10/20
483/483 [==============================] - 781s 2s/step - loss: 2.5249 - accuracy: 0.2513 - val_loss: 2.7312 - val_accuracy: 0.1891
Epoch 11/20
483/483 [==============================] - 793s 2s/step - loss: 2.5126 - accuracy: 0.2502 - val_loss: 2.7032 - val_accuracy: 0.2086
Epoch 12/20
483/483 [==============================] - 796s 2s/step - loss: 2.5071 - accuracy: 0.2574 - val_loss: 2.6935 - val_accuracy: 0.2125
Epoch 13/20
483/483 [==============================] - 796s 2s/step - loss: 2.4904 - accuracy: 0.2648 - val_loss: 2.6950 - val_accuracy: 0.2029
Epoch 14/20
483/483 [==============================] - 798s 2s/step - loss: 2.4629 - accuracy: 0.2666 - val_loss: 2.6804 - val_accuracy: 0.2156
Epoch 15/20
483/483 [==============================] - 796s 2s/step - loss: 2.4444 - accuracy: 0.2734 - val_loss: 2.6566 - val_accuracy: 0.2081
Epoch 16/20
483/483 [==============================] - 807s 2s/step - loss: 2.4286 - accuracy: 0.2785 - val_loss: 2.6724 - val_accuracy: 0.2188
Epoch 17/20
483/483 [==============================] - 825s 2s/step - loss: 2.4118 - accuracy: 0.2804 - val_loss: 2.6670 - val_accuracy: 0.2164
Epoch 18/20
483/483 [==============================] - 811s 2s/step - loss: 2.4023 - accuracy: 0.2842 - val_loss: 2.6788 - val_accuracy: 0.2045
Epoch 19/20
483/483 [==============================] - 803s 2s/step - loss: 2.3868 - accuracy: 0.2919 - val_loss: 2.6622 - val_accuracy: 0.2193
Epoch 20/20
483/483 [==============================] - 814s 2s/step - loss: 2.3757 - accuracy: 0.2879 - val_loss: 2.6858 - val_accuracy: 0.2206
Training accuracy: 0.2878885865211487
Validation accuracy: 0.22057677805423737
"""

## Augmented data console output:
"""
Epoch 1/20
1011/1011 [==============================] - 1546s 2s/step - loss: 3.0383 - accuracy: 0.0947 - val_loss: 2.9471 - val_accuracy: 0.1386
Epoch 2/20
1011/1011 [==============================] - 1168s 1s/step - loss: 2.9056 - accuracy: 0.1438 - val_loss: 2.8553 - val_accuracy: 0.1630
Epoch 3/20
1011/1011 [==============================] - 789s 781ms/step - loss: 2.8320 - accuracy: 0.1677 - val_loss: 2.8025 - val_accuracy: 0.1837
Epoch 4/20
1011/1011 [==============================] - 807s 798ms/step - loss: 2.7684 - accuracy: 0.1833 - val_loss: 2.7829 - val_accuracy: 0.1831
Epoch 5/20
1011/1011 [==============================] - 789s 780ms/step - loss: 2.7217 - accuracy: 0.1991 - val_loss: 2.7443 - val_accuracy: 0.1941
Epoch 6/20
1011/1011 [==============================] - 793s 784ms/step - loss: 2.6874 - accuracy: 0.2053 - val_loss: 2.7225 - val_accuracy: 0.1904
Epoch 7/20
1011/1011 [==============================] - 784s 775ms/step - loss: 2.6470 - accuracy: 0.2167 - val_loss: 2.6982 - val_accuracy: 0.2096
Epoch 8/20
1011/1011 [==============================] - 781s 772ms/step - loss: 2.6154 - accuracy: 0.2276 - val_loss: 2.6694 - val_accuracy: 0.2200
Epoch 9/20
1011/1011 [==============================] - 780s 772ms/step - loss: 2.5894 - accuracy: 0.2309 - val_loss: 2.6482 - val_accuracy: 0.2290
Epoch 10/20
1011/1011 [==============================] - 770s 761ms/step - loss: 2.5554 - accuracy: 0.2434 - val_loss: 2.6318 - val_accuracy: 0.2323
Epoch 11/20
1011/1011 [==============================] - 780s 771ms/step - loss: 2.5312 - accuracy: 0.2489 - val_loss: 2.6177 - val_accuracy: 0.2331
Epoch 12/20
1011/1011 [==============================] - 785s 776ms/step - loss: 2.5039 - accuracy: 0.2594 - val_loss: 2.5936 - val_accuracy: 0.2477
Epoch 13/20
1011/1011 [==============================] - 783s 775ms/step - loss: 2.4828 - accuracy: 0.2627 - val_loss: 2.6254 - val_accuracy: 0.2289
Epoch 14/20
1011/1011 [==============================] - 781s 772ms/step - loss: 2.4581 - accuracy: 0.2687 - val_loss: 2.5495 - val_accuracy: 0.2624
Epoch 15/20
1011/1011 [==============================] - 779s 771ms/step - loss: 2.4404 - accuracy: 0.2769 - val_loss: 2.5653 - val_accuracy: 0.2549
Epoch 16/20
1011/1011 [==============================] - 1069s 1s/step - loss: 2.4151 - accuracy: 0.2829 - val_loss: 2.5354 - val_accuracy: 0.2642
Epoch 17/20
1011/1011 [==============================] - 1512s 1s/step - loss: 2.3951 - accuracy: 0.2889 - val_loss: 2.5453 - val_accuracy: 0.2609
Epoch 18/20
1011/1011 [==============================] - 1634s 2s/step - loss: 2.3770 - accuracy: 0.2926 - val_loss: 2.5323 - val_accuracy: 0.2598
Epoch 19/20
1011/1011 [==============================] - 1700s 2s/step - loss: 2.3564 - accuracy: 0.2986 - val_loss: 2.4959 - val_accuracy: 0.2822
Epoch 20/20d
1011/1011 [==============================] - 1801s 2s/step - loss: 2.3345 - accuracy: 0.3047 - val_loss: 2.4823 - val_accuracy: 0.2808
Training accuracy: 0.30465707182884216
Validation accuracy: 0.2808125913143158
"""
