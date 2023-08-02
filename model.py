import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Image dimensions (adjust if needed)
img_width, img_height = 224, 224

# Batch size for training
batch_size = 32

# Prepare the data for training and testing
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Create a generator for the training data
train_generator = datagen.flow_from_directory(
    'dataset/training_set',  # Path to the training set
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  # Use a subset of the data for training
)

# Create a generator for the validation data
validation_generator = datagen.flow_from_directory(
    'dataset/training_set',  # Path to the training set
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Use a subset of the data for validation
)

# Load the pre-trained MobileNetV2 model without the top classification layer
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')

# Freeze the base model layers to prevent them from being trained
base_model.trainable = False

# Add a classification layer on top of the base model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
model.fit(train_generator, validation_data=validation_generator, epochs=epochs)

# Save the trained model
model.save('cats_dogs_classifier.h5')
