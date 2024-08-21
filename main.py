import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow.python.keras as keras
from PIL import Image
from pathlib import Path
import scipy
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

# print("Done with library declaration, Current version of Tensorflow is: ", tf.__version__)

# collect directory
data_dir = Path('Garbage/original_images')

transformer = T.Compose([T.Resize((32, 32)), T.ToTensor()])
# dataset = ImageFolder(data_dir, transform=transformer)
dataset = ImageFolder(root = "Garbage/original_images", transform=transformer)
# display class names
# print(dataset.classes)

###STEP3###
# display class distribution
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# counts = [393,491,400,584,472,127]
# ax.bar(dataset.classes,counts)
# plt.title('Class Distribution')
# plt.show()


# ###STEP4###
PATH_TEST = r"Garbage/original_images"
PATH_TRAIN = r"Garbage/processed_images"
class_names = ['cardboard', 'glass', 'metal','paper','plastic','trash']
imagepath_cardboard = r"Garbage/original_images/cardboard"
graypath_cardboard = r"Garbage/processed_images/cardboard"
Cardboard_listing = os.listdir(r"Garbage/original_images/cardboard")
for file in Cardboard_listing:
    im = Image.open(imagepath_cardboard + '/' + file)
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_cardboard + '/' + file, "JPEG")

imagepath_glass = r"Garbage/original_images/glass"
graypath_glass = r"Garbage/processed_images/glass"
Glass_listing = os.listdir(imagepath_glass)
for file in Glass_listing:
    im = Image.open(imagepath_glass + '/' + file)
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_glass + '/' + file, "JPEG")

imagepath_metal = r"Garbage/original_images/metal"
graypath_metal = r"Garbage/processed_images/metal"
Metal_listing = os.listdir(imagepath_metal)
for file in Metal_listing:
    im = Image.open(imagepath_metal + '/' + file)
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_metal + '/' + file, "JPEG")

imagepath_paper = r"Garbage/original_images/paper"
graypath_paper = r"Garbage/processed_images/paper"
Paper_listing = os.listdir(imagepath_paper)
for file in Paper_listing:
    im = Image.open(imagepath_paper + '/' + file)
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_paper + '/' + file, "JPEG")

imagepath_plastic = r"Garbage/original_images/plastic"
graypath_plastic = r"Garbage/processed_images/plastic"
Plastic_listing = os.listdir(imagepath_plastic)
for file in Plastic_listing:
    im = Image.open(imagepath_plastic + '/' + file)
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_plastic + '/' + file, "JPEG")

imagepath_trash = r"Garbage/original_images/trash"
graypath_trash = r"Garbage/processed_images/trash"
Trash_listing = os.listdir(imagepath_trash)
for file in Trash_listing:
    im = Image.open(imagepath_trash + '/' + file)
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_trash + '/' + file, "JPEG")

train_dir = os.path.join(PATH_TRAIN)
test_dir = os.path.join(PATH_TEST)

imagepath_cardboard_dir = os.path.join(imagepath_cardboard)
imagepath_glass_dir = os.path.join(imagepath_glass)
imagepath_metal_dir = os.path.join(imagepath_metal)
imagepath_paper_dir = os.path.join(imagepath_paper)
imagepath_plastic_dir = os.path.join(imagepath_plastic)
imagepath_trash_dir = os.path.join(imagepath_trash)

len(os.listdir(PATH_TRAIN))

IMG_HEIGHT = 32
IMG_WIDTH = 32

image_gen = ImageDataGenerator(rescale=1./255)

train_data_gen = image_gen.flow_from_directory(
    directory = train_dir,
    shuffle=True,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

test_data_gen = image_gen.flow_from_directory(
    directory = test_dir,
    shuffle=True,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')


# plt.figure()
# plt.imshow(sample_training_images[0])
# plt.show()
sample_data_gen = image_gen.flow_from_directory(
    directory = test_dir,
    shuffle=True,
    target_size = (200, 200),
    class_mode='categorical')

sample_training_images, _= next(sample_data_gen)
def plotImages(images_arr):
    fig, axes = plt.subplots(1,4, figsize=(30,30))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:4])
