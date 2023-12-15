import os
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# This is a simple MLP Classifier and serves as a benchmark to test against other model types. Note: this model is only run on the non-augmented combined data set due to computation capacity limitations.

# Load and preprocess the image data
data = '../../DermNet_Images/combined_v2'

X = []
y = []

count = 1
for category in os.listdir(data):
    category_dir = os.path.join(data, category)
    print('Working on category ', count)
    count+=1
    for image_file in os.listdir(category_dir):
        image_path = os.path.join(category_dir, image_file)
        image = io.imread(image_path)
        image = resize(image, (128, 128))
        X.append(image)
        y.append(category)


X = np.array(X)
y = np.array(y)

# Flatten the images
X = X.reshape(X.shape[0], -1)

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print('Done Preprocessing')

# Run 5-fold cross validation on this model on all the data
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

count = 0;
for train_index, test_index in kf.split(X, y_encoded):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    print("Creating model...")
    count+=1
    print(count)
    model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10, verbose=False)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Print the accuracies for each fold
for fold, accuracy in enumerate(accuracies):
    print(f'Fold {fold + 1} Accuracy: {accuracy:.2f}')

# Print the mean and standard deviation of the accuracies
print(f'Mean Accuracy: {np.mean(accuracies):.2f}')
print(f'Standard Deviation: {np.std(accuracies):.2f}')


# Console output:

# Fold 1 Accuracy: 0.09
# Fold 2 Accuracy: 0.09
# Fold 3 Accuracy: 0.11
# Fold 4 Accuracy: 0.09
# Fold 5 Accuracy: 0.09
# Mean Accuracy: 0.10
# Standard Deviation: 0.01