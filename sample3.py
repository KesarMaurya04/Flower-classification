import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('main.h5')

# Load a sample image
sample_image_path = 'C:\\Users\\Lenovo\\Desktop\\flower\\images\\dandelion1.jpg'
sample_image = load_img(sample_image_path, target_size=(150, 150))
sample_image = img_to_array(sample_image)
sample_image = np.expand_dims(sample_image, axis=0)
sample_image = sample_image / 255.0  # Rescale to [0, 1]

# Use the model to make predictions
predictions = model.predict(sample_image)
predicted_class = np.argmax(predictions[0])

# Define the class labels
class_labels = ['daisy', 'dandelion', 'roses', 'sunflowers','tulips']  # Define your class labels here

# Get the predicted class label
predicted_class_label = class_labels[predicted_class]

print("Predicted class:", predicted_class_label)