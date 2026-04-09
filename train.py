import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = 48

def load_images_from_directory(directory):
    images, labels = [], []
    label_map = {'ADHD-Hyperactive': 0, 'Typically Developing Children': 1}

    for class_name in os.listdir(directory):
        class_folder = os.path.join(directory, class_name)

        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                if filename.endswith((".jpg", ".png")):
                    img_path = os.path.join(class_folder, filename)

                    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                    img = img_to_array(img) / 255.0

                    images.append(img)
                    labels.append(label_map[class_name])

    return np.array(images), np.array(labels)


print("Loading data...")
xtrain, ytrain = load_images_from_directory("media/train")
xval, yval = load_images_from_directory("media/val")

print("Building model...")
base_model = VGG19(weights='imagenet', include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Training model...")
model.fit(xtrain, ytrain, epochs=5, validation_data=(xval, yval))

print("Saving model...")
model.save("media/adhd_model.h5")

print("✅ Training complete!")