import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os


def preprocess_image(image, target_size=(224, 224)):
    # Resize image
    image = cv2.resize(image, target_size)
    # Add channel dimension and normalize
    image = np.expand_dims(image, axis=-1)
    return image / 255.0


def create_training_data():
    images = []
    labels = []

    training_dir = "training"
    if not os.path.exists(training_dir):
        raise ValueError(f"Training directory '{training_dir}' does not exist")

    for filename in os.listdir(training_dir):
        if filename.endswith(".png"):
            # Load image
            img_path = os.path.join(training_dir, filename)
            # Update the annotation path to match your naming convention
            txt_path = img_path.replace(".png", ".txt")

            # Print debug info
            print(f"Looking for image: {img_path}")
            print(f"Looking for annotation: {txt_path}")

            if not os.path.exists(img_path):
                print(f"Warning: Image file {img_path} not found, skipping...")
                continue
            if not os.path.exists(txt_path):
                print(f"Warning: Annotation file {txt_path} not found, skipping...")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Failed to load image {img_path}, skipping...")
                continue

            # Rest of your existing code...
            processed_img = preprocess_image(img)

            try:
                with open(txt_path, "r") as f:
                    x, y, r = map(int, f.read().strip().split(","))
            except (ValueError, IOError) as e:
                print(f"Warning: Failed to parse annotation file {txt_path}: {e}")
                continue

            h, w = img.shape
            norm_annotation = [x / w, y / h, r / min(h, w)]

            images.append(processed_img)
            labels.append(norm_annotation)

    if not images:
        raise ValueError("No valid training data found")

    return np.array(images), np.array(labels)


def train_particle_detector():
    X, y = create_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(224, 224, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(3),  # [x, y, radius]
        ]
    )

    model.compile(
        optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"]
    )

    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    # Update save format to use the recommended .keras format
    model.save("particle_detector.keras")
    return model


if __name__ == "__main__":
    train_particle_detector()
