
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load the saved model
model = load_model('dog_cat_classifier3.h5')

# Load an image for prediction
img_path = 'path/to/image'

img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make predictions
prediction = model.predict(img_array)
if prediction[0] > 0.5:
    print('Dog')
else:
    print('Cat')
    
# Plot the image
plt.imshow(img)
plt.show()




