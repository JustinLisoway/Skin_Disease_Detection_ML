import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
import pandas as pd


# The 1st CNN model variation showed the most promising results. Thus, the parameters used to create that model are copied here and run again using a 5-fold cross validation. The model is only run on the augmented data due to the high computational cost. A summary of the results is provided in a comment at the end.

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

# Load filenames and labels
filenames = []
labels = []

for category in os.listdir(dataset_directory):
    category_path = os.path.join(dataset_directory, category)
    if os.path.isdir(category_path):
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            filenames.append(img_path)
            labels.append(category)


# Convert labels to integers
label_to_index = {label: i for i, label in enumerate(set(labels))}
labels = [label_to_index[label] for label in labels]


# Convert to numpy arrays
filenames = np.array(filenames)
labels = np.array(labels)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# For each fold, train and test the model on a different split
for fold, (train_indices, val_indices) in enumerate(skf.split(filenames, labels)):
    print(f"Fold {fold + 1}")
    train_dataframe = pd.DataFrame({"filename": filenames[train_indices], "label": labels[train_indices]})
    validation_dataframe = pd.DataFrame({"filename": filenames[val_indices], "label": labels[val_indices]})
    train_generator = data_generator.flow_from_dataframe(
        dataframe=train_dataframe,
        directory=None,
        x_col="filename",
        y_col="label",
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='raw',
        shuffle=True
    )
    validation_generator = data_generator.flow_from_dataframe(
        dataframe=validation_dataframe,
        directory=None,
        x_col="filename",
        y_col="label",
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='raw',
        shuffle=False
    )
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
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print("Validation accuracy:", val_accuracy)


## Augmented data console output:
'''Fold 1
Found 32328 validated image filenames.
Found 8083 validated image filenames.
Epoch 1/20
1011/1011 [==============================] - 872s 862ms/step - loss: 2.9741 - accuracy: 0.1195 - val_loss: 2.8718 - val_accuracy: 0.1486
Epoch 2/20
1011/1011 [==============================] - 954s 943ms/step - loss: 2.7747 - accuracy: 0.1808 - val_loss: 2.7448 - val_accuracy: 0.1888
Epoch 3/20
1011/1011 [==============================] - 959s 948ms/step - loss: 2.6626 - accuracy: 0.2149 - val_loss: 2.6539 - val_accuracy: 0.2185
Epoch 4/20
1011/1011 [==============================] - 1075s 1s/step - loss: 2.5700 - accuracy: 0.2428 - val_loss: 2.5701 - val_accuracy: 0.2443
Epoch 5/20
1011/1011 [==============================] - 756s 748ms/step - loss: 2.4846 - accuracy: 0.2667 - val_loss: 2.5074 - val_accuracy: 0.2592
Epoch 6/20
1011/1011 [==============================] - 940s 929ms/step - loss: 2.4163 - accuracy: 0.2873 - val_loss: 2.4608 - val_accuracy: 0.2792
Epoch 7/20
1011/1011 [==============================] - 906s 896ms/step - loss: 2.3620 - accuracy: 0.3030 - val_loss: 2.4327 - val_accuracy: 0.2883
Epoch 8/20
1011/1011 [==============================] - 746s 738ms/step - loss: 2.3088 - accuracy: 0.3168 - val_loss: 2.3807 - val_accuracy: 0.3008
Epoch 9/20
1011/1011 [==============================] - 1521s 2s/step - loss: 2.2609 - accuracy: 0.3346 - val_loss: 2.3664 - val_accuracy: 0.3035
Epoch 10/20
1011/1011 [==============================] - 771s 763ms/step - loss: 2.2288 - accuracy: 0.3409 - val_loss: 2.3528 - val_accuracy: 0.3104
Epoch 11/20
1011/1011 [==============================] - 908s 898ms/step - loss: 2.1913 - accuracy: 0.3523 - val_loss: 2.2877 - val_accuracy: 0.3281
Epoch 12/20
1011/1011 [==============================] - 1010s 999ms/step - loss: 2.1490 - accuracy: 0.3659 - val_loss: 2.2840 - val_accuracy: 0.3391
Epoch 13/20
1011/1011 [==============================] - 985s 974ms/step - loss: 2.1191 - accuracy: 0.3763 - val_loss: 2.2753 - val_accuracy: 0.3426
Epoch 14/20
1011/1011 [==============================] - 976s 965ms/step - loss: 2.0742 - accuracy: 0.3850 - val_loss: 2.2428 - val_accuracy: 0.3518
Epoch 15/20
1011/1011 [==============================] - 958s 947ms/step - loss: 2.0511 - accuracy: 0.3925 - val_loss: 2.2035 - val_accuracy: 0.3574
Epoch 16/20
1011/1011 [==============================] - 993s 982ms/step - loss: 2.0158 - accuracy: 0.4020 - val_loss: 2.2231 - val_accuracy: 0.3600
Epoch 17/20
1011/1011 [==============================] - 930s 919ms/step - loss: 1.9850 - accuracy: 0.4140 - val_loss: 2.1505 - val_accuracy: 0.3735
Epoch 18/20
1011/1011 [==============================] - 885s 875ms/step - loss: 1.9674 - accuracy: 0.4186 - val_loss: 2.1530 - val_accuracy: 0.3797
Epoch 19/20
1011/1011 [==============================] - 922s 912ms/step - loss: 1.9320 - accuracy: 0.4264 - val_loss: 2.1434 - val_accuracy: 0.3742
Epoch 20/20
1011/1011 [==============================] - 877s 868ms/step - loss: 1.9144 - accuracy: 0.4324 - val_loss: 2.1438 - val_accuracy: 0.3859
253/253 [==============================] - 93s 368ms/step - loss: 2.1229 - accuracy: 0.3810
Validation accuracy: 0.3810466527938843
Fold 2
Found 32329 validated image filenames.
Found 8082 validated image filenames.
Epoch 1/20
1011/1011 [==============================] - 858s 848ms/step - loss: 3.0024 - accuracy: 0.1069 - val_loss: 2.8934 - val_accuracy: 0.1387
Epoch 2/20
1011/1011 [==============================] - 838s 828ms/step - loss: 2.7912 - accuracy: 0.1780 - val_loss: 2.7237 - val_accuracy: 0.1930
Epoch 3/20
1011/1011 [==============================] - 831s 821ms/step - loss: 2.6544 - accuracy: 0.2196 - val_loss: 2.6180 - val_accuracy: 0.2279
Epoch 4/20
1011/1011 [==============================] - 981s 970ms/step - loss: 2.5565 - accuracy: 0.2464 - val_loss: 2.5682 - val_accuracy: 0.2414
Epoch 5/20
1011/1011 [==============================] - 789s 780ms/step - loss: 2.4696 - accuracy: 0.2729 - val_loss: 2.4908 - val_accuracy: 0.2689
Epoch 6/20
1011/1011 [==============================] - 758s 750ms/step - loss: 2.3933 - accuracy: 0.2920 - val_loss: 2.4383 - val_accuracy: 0.2902
Epoch 7/20
1011/1011 [==============================] - 777s 769ms/step - loss: 2.3192 - accuracy: 0.3184 - val_loss: 2.4344 - val_accuracy: 0.2941
Epoch 8/20
1011/1011 [==============================] - 817s 808ms/step - loss: 2.2529 - accuracy: 0.3404 - val_loss: 2.3927 - val_accuracy: 0.3057
Epoch 9/20
1011/1011 [==============================] - 886s 876ms/step - loss: 2.2020 - accuracy: 0.3547 - val_loss: 2.3407 - val_accuracy: 0.3191
Epoch 10/20
1011/1011 [==============================] - 910s 900ms/step - loss: 2.1452 - accuracy: 0.3670 - val_loss: 2.3143 - val_accuracy: 0.3206
Epoch 11/20
1011/1011 [==============================] - 883s 873ms/step - loss: 2.0962 - accuracy: 0.3810 - val_loss: 2.2839 - val_accuracy: 0.3364
Epoch 12/20
1011/1011 [==============================] - 883s 873ms/step - loss: 2.0593 - accuracy: 0.3914 - val_loss: 2.2271 - val_accuracy: 0.3587
Epoch 13/20
1011/1011 [==============================] - 872s 863ms/step - loss: 2.0077 - accuracy: 0.4092 - val_loss: 2.2127 - val_accuracy: 0.3627
Epoch 14/20
1011/1011 [==============================] - 865s 855ms/step - loss: 1.9681 - accuracy: 0.4188 - val_loss: 2.1770 - val_accuracy: 0.3726
Epoch 15/20
1011/1011 [==============================] - 874s 865ms/step - loss: 1.9350 - accuracy: 0.4309 - val_loss: 2.1594 - val_accuracy: 0.3759
Epoch 16/20
1011/1011 [==============================] - 874s 864ms/step - loss: 1.8967 - accuracy: 0.4406 - val_loss: 2.1250 - val_accuracy: 0.3900
Epoch 17/20
1011/1011 [==============================] - 874s 865ms/step - loss: 1.8750 - accuracy: 0.4467 - val_loss: 2.2039 - val_accuracy: 0.3703
Epoch 18/20
1011/1011 [==============================] - 877s 867ms/step - loss: 1.8362 - accuracy: 0.4576 - val_loss: 2.1188 - val_accuracy: 0.3966
Epoch 19/20
1011/1011 [==============================] - 878s 869ms/step - loss: 1.8132 - accuracy: 0.4650 - val_loss: 2.0993 - val_accuracy: 0.4024
Epoch 20/20
1011/1011 [==============================] - 874s 864ms/step - loss: 1.7750 - accuracy: 0.4742 - val_loss: 2.0998 - val_accuracy: 0.4070
253/253 [==============================] - 94s 372ms/step - loss: 2.1010 - accuracy: 0.4018
Validation accuracy: 0.40175700187683105
Fold 3
Found 32329 validated image filenames.
Found 8082 validated image filenames.
Epoch 1/20
1011/1011 [==============================] - 878s 868ms/step - loss: 2.9945 - accuracy: 0.1089 - val_loss: 2.8834 - val_accuracy: 0.1466
Epoch 2/20
1011/1011 [==============================] - 875s 866ms/step - loss: 2.8171 - accuracy: 0.1665 - val_loss: 2.8181 - val_accuracy: 0.1743
Epoch 3/20
1011/1011 [==============================] - 866s 856ms/step - loss: 2.7066 - accuracy: 0.1982 - val_loss: 2.6848 - val_accuracy: 0.2082
Epoch 4/20
1011/1011 [==============================] - 866s 856ms/step - loss: 2.6032 - accuracy: 0.2322 - val_loss: 2.5702 - val_accuracy: 0.2418
Epoch 5/20
1011/1011 [==============================] - 872s 863ms/step - loss: 2.5310 - accuracy: 0.2537 - val_loss: 2.5314 - val_accuracy: 0.2606
Epoch 6/20
1011/1011 [==============================] - 876s 867ms/step - loss: 2.4550 - accuracy: 0.2748 - val_loss: 2.4737 - val_accuracy: 0.2798
Epoch 7/20
1011/1011 [==============================] - 866s 857ms/step - loss: 2.3929 - accuracy: 0.2925 - val_loss: 2.4321 - val_accuracy: 0.2900
Epoch 8/20
1011/1011 [==============================] - 862s 853ms/step - loss: 2.3331 - accuracy: 0.3100 - val_loss: 2.4607 - val_accuracy: 0.2898
Epoch 9/20
1011/1011 [==============================] - 861s 851ms/step - loss: 2.2826 - accuracy: 0.3269 - val_loss: 2.3627 - val_accuracy: 0.3132
Epoch 10/20
1011/1011 [==============================] - 865s 855ms/step - loss: 2.2328 - accuracy: 0.3412 - val_loss: 2.3581 - val_accuracy: 0.3113
Epoch 11/20
1011/1011 [==============================] - 871s 861ms/step - loss: 2.1971 - accuracy: 0.3499 - val_loss: 2.3402 - val_accuracy: 0.3244
Epoch 12/20
1011/1011 [==============================] - 866s 857ms/step - loss: 2.1577 - accuracy: 0.3633 - val_loss: 2.3211 - val_accuracy: 0.3327
Epoch 13/20
1011/1011 [==============================] - 860s 850ms/step - loss: 2.1176 - accuracy: 0.3746 - val_loss: 2.2308 - val_accuracy: 0.3473
Epoch 14/20
1011/1011 [==============================] - 859s 849ms/step - loss: 2.0852 - accuracy: 0.3828 - val_loss: 2.2452 - val_accuracy: 0.3473
Epoch 15/20
1011/1011 [==============================] - 871s 861ms/step - loss: 2.0550 - accuracy: 0.3949 - val_loss: 2.2347 - val_accuracy: 0.3578
Epoch 16/20
1011/1011 [==============================] - 871s 861ms/step - loss: 2.0240 - accuracy: 0.4034 - val_loss: 2.2396 - val_accuracy: 0.3628
Epoch 17/20
1011/1011 [==============================] - 862s 853ms/step - loss: 1.9980 - accuracy: 0.4108 - val_loss: 2.2066 - val_accuracy: 0.3651
Epoch 18/20
1011/1011 [==============================] - 856s 847ms/step - loss: 1.9686 - accuracy: 0.4162 - val_loss: 2.1533 - val_accuracy: 0.3868
Epoch 19/20
1011/1011 [==============================] - 857s 847ms/step - loss: 1.9434 - accuracy: 0.4268 - val_loss: 2.2152 - val_accuracy: 0.3655
Epoch 20/20
1011/1011 [==============================] - 858s 848ms/step - loss: 1.9196 - accuracy: 0.4306 - val_loss: 2.1438 - val_accuracy: 0.3817
253/253 [==============================] - 92s 365ms/step - loss: 2.1670 - accuracy: 0.3831
Validation accuracy: 0.38307350873947144
Fold 4
Found 32329 validated image filenames.
Found 8082 validated image filenames.
Epoch 1/20
1011/1011 [==============================] - 861s 852ms/step - loss: 2.9830 - accuracy: 0.1178 - val_loss: 2.8684 - val_accuracy: 0.1553
Epoch 2/20
1011/1011 [==============================] - 866s 856ms/step - loss: 2.7566 - accuracy: 0.1884 - val_loss: 2.7164 - val_accuracy: 0.1933
Epoch 3/20
1011/1011 [==============================] - 864s 854ms/step - loss: 2.6262 - accuracy: 0.2273 - val_loss: 2.5928 - val_accuracy: 0.2392
Epoch 4/20
1011/1011 [==============================] - 859s 850ms/step - loss: 2.5230 - accuracy: 0.2547 - val_loss: 2.5316 - val_accuracy: 0.2606
Epoch 5/20
1011/1011 [==============================] - 861s 852ms/step - loss: 2.4420 - accuracy: 0.2805 - val_loss: 2.4761 - val_accuracy: 0.2762
Epoch 6/20
1011/1011 [==============================] - 837s 828ms/step - loss: 2.3474 - accuracy: 0.3089 - val_loss: 2.4434 - val_accuracy: 0.2861
Epoch 7/20
1011/1011 [==============================] - 802s 794ms/step - loss: 2.2815 - accuracy: 0.3294 - val_loss: 2.3483 - val_accuracy: 0.3097
Epoch 8/20
1011/1011 [==============================] - 850s 840ms/step - loss: 2.2006 - accuracy: 0.3508 - val_loss: 2.3034 - val_accuracy: 0.3310
Epoch 9/20
1011/1011 [==============================] - 845s 836ms/step - loss: 2.1394 - accuracy: 0.3716 - val_loss: 2.2872 - val_accuracy: 0.3396
Epoch 10/20
1011/1011 [==============================] - 907s 897ms/step - loss: 2.0721 - accuracy: 0.3907 - val_loss: 2.2284 - val_accuracy: 0.3503
Epoch 11/20
1011/1011 [==============================] - 973s 963ms/step - loss: 2.0193 - accuracy: 0.4073 - val_loss: 2.2440 - val_accuracy: 0.3519
Epoch 12/20
1011/1011 [==============================] - 970s 960ms/step - loss: 1.9694 - accuracy: 0.4231 - val_loss: 2.1616 - val_accuracy: 0.3703
Epoch 13/20
1011/1011 [==============================] - 974s 963ms/step - loss: 1.9221 - accuracy: 0.4358 - val_loss: 2.1884 - val_accuracy: 0.3680
Epoch 14/20
1011/1011 [==============================] - 958s 948ms/step - loss: 1.8814 - accuracy: 0.4464 - val_loss: 2.1164 - val_accuracy: 0.3945
Epoch 15/20
1011/1011 [==============================] - 1004s 993ms/step - loss: 1.8436 - accuracy: 0.4597 - val_loss: 2.1357 - val_accuracy: 0.3862
Epoch 16/20
1011/1011 [==============================] - 918s 908ms/step - loss: 1.8102 - accuracy: 0.4693 - val_loss: 2.1078 - val_accuracy: 0.4065
Epoch 17/20
1011/1011 [==============================] - 944s 933ms/step - loss: 1.7727 - accuracy: 0.4785 - val_loss: 2.0956 - val_accuracy: 0.4067
Epoch 18/20
1011/1011 [==============================] - 999s 988ms/step - loss: 1.7315 - accuracy: 0.4869 - val_loss: 2.0576 - val_accuracy: 0.4248
Epoch 19/20
1011/1011 [==============================] - 889s 879ms/step - loss: 1.7068 - accuracy: 0.4967 - val_loss: 2.0570 - val_accuracy: 0.4220
Epoch 20/20
1011/1011 [==============================] - 892s 883ms/step - loss: 1.6899 - accuracy: 0.5006 - val_loss: 2.0634 - val_accuracy: 0.4227
253/253 [==============================] - 111s 438ms/step - loss: 2.0583 - accuracy: 0.4266
Validation accuracy: 0.4266270697116852
Fold 5
Found 32329 validated image filenames.
Found 8082 validated image filenames.
Epoch 1/20
1011/1011 [==============================] - 899s 889ms/step - loss: 3.0004 - accuracy: 0.1113 - val_loss: 2.8610 - val_accuracy: 0.1500
Epoch 2/20
1011/1011 [==============================] - 870s 860ms/step - loss: 2.7841 - accuracy: 0.1822 - val_loss: 2.7484 - val_accuracy: 0.1915
Epoch 3/20
1011/1011 [==============================] - 932s 922ms/step - loss: 2.6418 - accuracy: 0.2205 - val_loss: 2.6042 - val_accuracy: 0.2329
Epoch 4/20
1011/1011 [==============================] - 909s 900ms/step - loss: 2.5453 - accuracy: 0.2470 - val_loss: 2.5176 - val_accuracy: 0.2633
Epoch 5/20
1011/1011 [==============================] - 895s 885ms/step - loss: 2.4593 - accuracy: 0.2737 - val_loss: 2.4619 - val_accuracy: 0.2784
Epoch 6/20
1011/1011 [==============================] - 882s 872ms/step - loss: 2.3803 - accuracy: 0.2984 - val_loss: 2.4521 - val_accuracy: 0.2830
Epoch 7/20
1011/1011 [==============================] - 903s 893ms/step - loss: 2.3178 - accuracy: 0.3169 - val_loss: 2.3901 - val_accuracy: 0.3024
Epoch 8/20
1011/1011 [==============================] - 857s 847ms/step - loss: 2.2506 - accuracy: 0.3382 - val_loss: 2.3435 - val_accuracy: 0.3190
Epoch 9/20
1011/1011 [==============================] - 834s 825ms/step - loss: 2.1905 - accuracy: 0.3580 - val_loss: 2.3327 - val_accuracy: 0.3155
Epoch 10/20
1011/1011 [==============================] - 910s 900ms/step - loss: 2.1370 - accuracy: 0.3724 - val_loss: 2.2874 - val_accuracy: 0.3352
Epoch 11/20
1011/1011 [==============================] - 801s 792ms/step - loss: 2.0927 - accuracy: 0.3827 - val_loss: 2.2574 - val_accuracy: 0.3445
Epoch 12/20
1011/1011 [==============================] - 856s 846ms/step - loss: 2.0440 - accuracy: 0.4002 - val_loss: 2.2236 - val_accuracy: 0.3607
Epoch 13/20
1011/1011 [==============================] - 843s 834ms/step - loss: 2.0058 - accuracy: 0.4116 - val_loss: 2.1942 - val_accuracy: 0.3679
Epoch 14/20
1011/1011 [==============================] - 840s 830ms/step - loss: 1.9626 - accuracy: 0.4234 - val_loss: 2.1997 - val_accuracy: 0.3683
Epoch 15/20
1011/1011 [==============================] - 821s 811ms/step - loss: 1.9316 - accuracy: 0.4309 - val_loss: 2.1831 - val_accuracy: 0.3729
Epoch 16/20
1011/1011 [==============================] - 821s 812ms/step - loss: 1.9005 - accuracy: 0.4365 - val_loss: 2.1491 - val_accuracy: 0.3839
Epoch 17/20
1011/1011 [==============================] - 823s 814ms/step - loss: 1.8636 - accuracy: 0.4531 - val_loss: 2.1393 - val_accuracy: 0.3928
Epoch 18/20
1011/1011 [==============================] - 823s 814ms/step - loss: 1.8269 - accuracy: 0.4573 - val_loss: 2.1410 - val_accuracy: 0.3869
Epoch 19/20
1011/1011 [==============================] - 821s 812ms/step - loss: 1.8118 - accuracy: 0.4649 - val_loss: 2.1072 - val_accuracy: 0.4008
Epoch 20/20
1011/1011 [==============================] - 828s 819ms/step - loss: 1.7723 - accuracy: 0.4783 - val_loss: 2.0758 - val_accuracy: 0.4077
253/253 [==============================] - 90s 355ms/step - loss: 2.0556 - accuracy: 0.4162
Validation accuracy: 0.4162335991859436
'''

'''
Summary:

Fold 1: 0.3810466527938843
Fold 2: 0.40175700187683105
Fold 3: 0.38307350873947144
Fold 4: 0.4266270697116852
Fold 5: 0.4162335991859436

Average: 0.40174756646156
'''