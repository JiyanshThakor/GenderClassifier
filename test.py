import joblib
import numpy as np
from PIL import Image

model = joblib.load('GenderClassifier.pkl')
test_face = np.array(Image.open('demo.jpg').resize((32,32))).flatten().reshape(1,-1)
prediction = model.predict(test_face)
if prediction == 0:
    print("Male")
else:
    print("Female")