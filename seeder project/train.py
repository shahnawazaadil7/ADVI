import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import ResNet50, MobileNetV2
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import cv2
from PIL import Image, UnidentifiedImageError

# Define dataset path
dataset_path = "/Users/shahnawazaadil/Desktop/Github/ADVI/seeder project/paddy images"

# Ensure output directory exists
os.makedirs("models", exist_ok=True)

# Data Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='binary', subset='training')
val_data = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='binary', subset='validation')

# CNN Model
def train_cnn():
    try:
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_data, validation_data=val_data, epochs=10)
        model.save('models/cnn_model.keras')  # Changed to .keras
    except Exception as e:
        print(f"Error training CNN model: {e}")

def train_resnet_rf():
    try:
        print("Initializing ResNet50 feature extractor...")
        feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        images, labels = [], []
        classes = ['paddy with pests', 'paddy without pests']
        
        for label, class_name in enumerate(classes):
            class_path = os.path.join(dataset_path, class_name)
            print(f"Processing class: {class_name}")
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')  # Convert to RGB
                    img = img.resize((224, 224))
                    img = np.array(img)
                    img = preprocess_input(img)
                    images.append(img)
                    labels.append(label)
                except UnidentifiedImageError:
                    print(f"Unidentified image error for file: {img_path}")
                except Exception as e:
                    print(f"Error processing file {img_path}: {e}")
        
        images, labels = np.array(images), np.array(labels)
        
        features = feature_extractor.predict(images)
    
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        train_accuracy = rf_model.score(X_train, y_train)
        test_accuracy = rf_model.score(X_test, y_test)
        
        print("Saving Random Forest model...")
        joblib.dump(rf_model, 'models/rf_model.pkl')
        print("ResNet + Random Forest model training completed successfully!")
    except Exception as e:
        print(f"Error training ResNet + Random Forest model: {e}")

# MobileNetV2 Model
def train_mobilenet():
    try:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3), pooling='avg')
        model = keras.Sequential([
            base_model,
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_data, validation_data=val_data, epochs=10)
        model.save('models/mobilenet_model.keras')  # Changed to .keras
    except Exception as e:
        print(f"Error training MobileNetV2 model: {e}")

# Train all models
if not os.path.exists('models'):
    os.makedirs('models')

# train_cnn()
train_resnet_rf()
# train_mobilenet()
print("Training completed and models saved!")