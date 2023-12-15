import os
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# Random image augmentation function
def random_augmentation(image):
    choice = random.choice([1, 2, 3, 4])
    if choice == 1:
        # Randomly adjust brightness and contrast
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.7, 1.3))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.7, 1.3))
    elif choice == 2:
        # Randomly add noise
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0, 2)))
    elif choice == 3:
        # Randomly flip horizontally or vertically
        if random.choice([True, False]):
            image = ImageOps.mirror(image)
        if random.choice([True, False]):
            image = ImageOps.flip(image)
    else:
        # Randomly rotate the image
        angle = random.uniform(-20, 20)
        image = image.rotate(angle)
    return image


# Find the folder with the most images
root_dir = "../../../DermNet_Images/combined_v2"  # Replace with your dataset folder
max_images = 0
most_images_folder = None

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        num_images = len(os.listdir(folder_path))
        if num_images > max_images:
            max_images = num_images
            most_images_folder = folder

# Loop through each folder and generate image augmentations randomly
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)

    # Skip non-folders and largest folder
    if not os.path.isdir(folder_path) or folder == most_images_folder:
        continue

    num_images = len(os.listdir(folder_path))

    # Generate augmented images until the folder has as many images as the largest folder
    while num_images < max_images:
        image_files = os.listdir(folder_path)
        source_image = os.path.join(folder_path, random.choice(image_files))
        new_image = os.path.join(folder_path, f"new_image_{num_images + 1}.jpg")

        # Open the source image
        img = Image.open(source_image)

        # Apply random augmentation
        img = random_augmentation(img)

        # Save the augmented image
        img.save(new_image)
        num_images += 1

# Now, all folders should have the same number of images (max_images) with randomized augmentations.
# Use test script to count images
