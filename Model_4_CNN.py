from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# This is the 4th CNN model variation and was only run on the augmented data.

# Load data and preprocess
dataset_directory = '../../DermNet_Images/combined_v2'
input_shape = (224, 224, 3)
num_classes = 23
batch_size = 32
epochs = 20

data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
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

# # Create model with altered parameters
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
    layers.Dropout(0.7),  # Increase dropout rate
    layers.Dense(num_classes, activation='softmax')
])

# Also try compiling the model with an even lower learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Run the model on whole dataset
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, batch_size=64, callbacks=[early_stopping])

print("Training accuracy:", history.history['accuracy'][-1])
print("Validation accuracy:", history.history['val_accuracy'][-1])

## Augmented data console output:
"""
Epoch 1/20
1011/1011 [==============================] - 4649s 5s/step - loss: 3.1105 - accuracy: 0.0674 - val_loss: 3.0464 - val_accuracy: 0.1034
Epoch 2/20
1011/1011 [==============================] - 3892s 4s/step - loss: 3.0258 - accuracy: 0.1039 - val_loss: 2.9686 - val_accuracy: 0.1425
Epoch 3/20
1011/1011 [==============================] - 3004s 3s/step - loss: 2.9753 - accuracy: 0.1231 - val_loss: 2.9245 - val_accuracy: 0.1390
Epoch 4/20
1011/1011 [==============================] - 3050s 3s/step - loss: 2.9420 - accuracy: 0.1342 - val_loss: 2.9079 - val_accuracy: 0.1515
Epoch 5/20
1011/1011 [==============================] - 2926s 3s/step - loss: 2.9172 - accuracy: 0.1410 - val_loss: 2.9023 - val_accuracy: 0.1564
Epoch 6/20
1011/1011 [==============================] - 2980s 3s/step - loss: 2.8934 - accuracy: 0.1506 - val_loss: 2.8728 - val_accuracy: 0.1616
Epoch 7/20
1011/1011 [==============================] - 3064s 3s/step - loss: 2.8735 - accuracy: 0.1543 - val_loss: 2.8517 - val_accuracy: 0.1687
Epoch 8/20
1011/1011 [==============================] - 2885s 3s/step - loss: 2.8532 - accuracy: 0.1637 - val_loss: 2.8519 - val_accuracy: 0.1697
Epoch 9/20
1011/1011 [==============================] - 2938s 3s/step - loss: 2.8380 - accuracy: 0.1695 - val_loss: 2.8305 - val_accuracy: 0.1724
Epoch 10/20
1011/1011 [==============================] - 2956s 3s/step - loss: 2.8269 - accuracy: 0.1701 - val_loss: 2.8202 - val_accuracy: 0.1812
Epoch 11/20
1011/1011 [==============================] - 3077s 3s/step - loss: 2.8143 - accuracy: 0.1750 - val_loss: 2.8117 - val_accuracy: 0.1898
Epoch 12/20
1011/1011 [==============================] - 2998s 3s/step - loss: 2.8009 - accuracy: 0.1770 - val_loss: 2.7997 - val_accuracy: 0.1836
Epoch 13/20
1011/1011 [==============================] - 2932s 3s/step - loss: 2.7888 - accuracy: 0.1839 - val_loss: 2.7844 - val_accuracy: 0.1870
Epoch 14/20
1011/1011 [==============================] - 2855s 3s/step - loss: 2.7700 - accuracy: 0.1861 - val_loss: 2.7696 - val_accuracy: 0.1977
Epoch 15/20
1011/1011 [==============================] - 2733s 3s/step - loss: 2.7594 - accuracy: 0.1881 - val_loss: 2.7735 - val_accuracy: 0.1872
Epoch 16/20
1011/1011 [==============================] - 2677s 3s/step - loss: 2.7488 - accuracy: 0.1944 - val_loss: 2.7569 - val_accuracy: 0.1971
Epoch 17/20
1011/1011 [==============================] - 2761s 3s/step - loss: 2.7388 - accuracy: 0.1962 - val_loss: 2.7533 - val_accuracy: 0.1996
Epoch 18/20
1011/1011 [==============================] - 2749s 3s/step - loss: 2.7294 - accuracy: 0.1995 - val_loss: 2.7656 - val_accuracy: 0.1895
Epoch 19/20
1011/1011 [==============================] - 2679s 3s/step - loss: 2.7150 - accuracy: 0.2027 - val_loss: 2.7327 - val_accuracy: 0.2049
Epoch 20/20
1011/1011 [==============================] - 2702s 3s/step - loss: 2.7066 - accuracy: 0.2044 - val_loss: 2.7282 - val_accuracy: 0.2023
Training accuracy: 0.20440348982810974
Validation accuracy: 0.202279195189476
"""