import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, Model
import os

print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Dataset paths
train_dir = "datasets/cats&dogs/train"
val_dir = "datasets/cats&dogs/validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


# Training Data Generator (with augmentation)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation Data Generator
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


# Load training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Load validation data
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)


# Load pretrained EfficientNet
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False


# Custom classifier head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)


# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Print model summary
model.summary()


# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)


# Save model
os.makedirs("models", exist_ok=True)
model.save("models/cat_dog_model.h5")

print("Model saved successfully!")
