from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.transform import resize 
import os
import joblib

x_train, y_train = [], []

for folder in ['Training/male', 'Training/female']:
    label = 0 if os.path.basename(folder) == 'male' else 1
    for img_file in os.listdir(folder):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')) or 'jpg' in img_file.lower():
            img_path = os.path.join(folder, img_file)
            img = Image.open(img_path)
            img_resized = img.resize((32,32))
            img_array = np.array(img_resized).flatten()
            x_train.append(img_array)
            y_train.append(label)

x_val, y_val = [], []

for folder in ['Validation/male', 'Validation/female']:
    label = 0 if os.path.basename(folder) == 'male' else 1
    for img_file in os.listdir(folder):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')) or 'jpg' in img_file.lower():
            img_path = os.path.join(folder, img_file)
            img = Image.open(img_path)
            img_resized = img.resize((32,32))
            img_array = np.array(img_resized).flatten()
            x_val.append(img_array)
            y_val.append(label)

model = RandomForestClassifier(n_estimators=10)
x_train_flat = np.array(x_train)
x_val_flat = np.array(x_val)

print("=== DEBUG ===")
print("Male train images:", len([f for f in os.listdir('Training/male') if 'jpg' in f.lower()]))
print("Female train images:", len([f for f in os.listdir('Training/female') if 'jpg' in f.lower()]))
print("y_train unique labels:", set(y_train))
print("y_train length:", len(y_train))
print("==============")

model.fit(x_train_flat, y_train)
print(accuracy_score(y_val, model.predict(x_val_flat)))

joblib.dump(model, "GenderClassifier.pkl")