import os
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('cats_dogs_classifier.h5')

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    # Load the image from the file and resize it to the target size
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    # Convert the image to a numpy array and expand the dimensions to match the model input shape
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # Normalize the pixel values to the range [0, 1]
    image /= 255.0
    return image

# Function to predict whether the image contains a cat or a dog
def predict_image(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)
    # Make a prediction using the loaded model
    prediction = model.predict(image)[0][0]
    # If the prediction score is greater than or equal to 0.5, classify as 'dog', otherwise as 'cat'
    if prediction >= 0.5:
        return 'dog'
    else:
        return 'cat'

# Test all cat images from cat.4001.jpg to cat.5000.jpg and count predictions
if __name__ == "__main__":
    # Path to the folder containing the cat images to be tested
    test_image_folder = 'dataset/test_set/cats/'
    cat_count = 0
    dog_count = 0

    # Loop through the image filenames from cat.4001.jpg to cat.5000.jpg
    for i in range(4001, 5001):
        image_filename = f'cat.{i}.jpg'
        sample_image_path = os.path.join(test_image_folder, image_filename)
        # Predict whether the image contains a cat or a dog
        predicted_class = predict_image(sample_image_path)

        # Update the counts based on the prediction
        if predicted_class == 'dog':
            dog_count += 1
        else:
            cat_count += 1

        # Print the individual prediction and the current counts
        print(f"{image_filename} is predicted to be a {predicted_class}.")
        print(f"Total number of cat predictions: {cat_count}")
        print(f"Total number of dog predictions: {dog_count}")
