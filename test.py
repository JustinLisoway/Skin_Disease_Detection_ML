import os

balanced = '../../../DermNet_Images/combined_v2'
balanced_dirs = os.listdir(balanced)

count = 0

for category in balanced_dirs:
    image_count = 0
    category_dir = os.path.join(balanced, category)
    images = os.listdir(category_dir)
    for image in images:
        count+=1
        image_count+=1
print(count)

combined = '../../../DermNet_Images/combined'
balanced_dirs = os.listdir(combined)

count = 0

for category in balanced_dirs:
    image_count = 0
    category_dir = os.path.join(combined, category)
    images = os.listdir(category_dir)
    for image in images:
        count+=1
        image_count+=1
print(count)


test = '../../../DermNet_Images/test'
train = '../../../DermNet_Images/train'

test_dir = os.listdir(test)
train_dir = os.listdir(train)

count_test = 0
count_train = 0
test_dir.remove('.DS_Store')
train_dir.remove('.DS_Store')

for category in test_dir:
    image_count = 0
    category_dir = os.path.join(test, category)
    images = os.listdir(category_dir)
    for image in images:
        count_test+=1
        image_count+=1

for category in train_dir:
    image_count = 0
    category_dir = os.path.join(train, category)
    images = os.listdir(category_dir)
    for image in images:
        count_train+=1
        image_count+=1

print(count_test)
print(count_train)
total = count_test + count_train
print(total)