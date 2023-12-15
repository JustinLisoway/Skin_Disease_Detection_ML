import os
import shutil

original_test_dataset_dir = '../../../DermNet_Images/test'
original_train_dataset_dir = '../../../DermNet_Images/train'

original_test_dirs = os.listdir(original_test_dataset_dir)
original_test_dirs.remove('.DS_Store')
original_train_dirs = os.listdir(original_train_dataset_dir)
original_train_dirs.remove('.DS_Store')

os.makedirs('../../../DermNet_Images/combined', exist_ok=True)
combined_path = '../../../DermNet_Images/combined'

# Combine Test and Train into one set

for category in original_test_dirs:
    category_dir = os.path.join(original_test_dataset_dir, category)
    images = os.listdir(category_dir)
    new_dir = os.path.join(combined_path, category)
    os.makedirs(new_dir, exist_ok=True)
    for image in images:
        image_src = os.path.join(category_dir, image)
        shutil.copy(image_src, new_dir)

for category in original_train_dirs:
    category_dir = os.path.join(original_train_dataset_dir, category)
    images = os.listdir(category_dir)
    new_dir = os.path.join(combined_path, category)
    os.makedirs(new_dir, exist_ok=True)
    for image in images:
        image_src = os.path.join(category_dir, image)
        shutil.copy(image_src, new_dir)



# Count Images

new_combined_dataset_dir = '../../../DermNet_Images/combined'

new_combined_dirs = os.listdir(new_combined_dataset_dir)

test_count = 0
train_count = 0
combined_count = 0

for category in original_test_dirs:
    category_dir = os.path.join(original_test_dataset_dir, category)
    images = os.listdir(category_dir)
    for image in images:
        test_count+=1

for category in original_train_dirs:
    category_dir = os.path.join(original_train_dataset_dir, category)
    images = os.listdir(category_dir)
    for image in images:
        train_count+=1

for category in new_combined_dirs:
    image_count = 0
    category_dir = os.path.join(new_combined_dataset_dir, category)
    images = os.listdir(category_dir)
    print(category)
    for image in images:
        image_count += 1
        combined_count+=1
    print(image_count)

print(test_count+train_count)
print(test_count+train_count-combined_count)

# Removed 270 duplicate images

