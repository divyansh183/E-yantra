import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np

# Define data directories
data_dir = "C:/Users/arnav/Downloads/training-20231027T075137Z-001/training"

# Define image size and batch size
image_size = (224, 224)
batch_size = 32

# Create data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values
    rotation_range=20,  # Data augmentation: random rotation
    width_shift_range=0.2,  # Data augmentation: random horizontal shift
    height_shift_range=0.2,  # Data augmentation: random vertical shift
    horizontal_flip=True,  # Data augmentation: horizontal flip
    validation_split=0.3  # Split a portion for validation
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Make sure it's set to categorical
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Make sure it's set to categorical
    subset='validation'
)

# Define the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)  # 5 classes for event classification
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Save the trained model
model.save("event_classification_model.h5")
