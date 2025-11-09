import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
dataset_path = os.path.expanduser("~/datasets/flower_photos") 

if not os.path.exists(dataset_path):
    dataset_path = tf.keras.utils.get_file("flower_photos", dataset_url, untar=True, cache_dir="~/datasets")

model = tf.keras.models.load_model("flower_classifier.h5")
print("✅ Model loaded successfully!")

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(dataset_path), 
    image_size=(224, 224),
    batch_size=32,
    label_mode="int"
)

class_names = test_dataset.class_names

normalization_layer = tf.keras.layers.Rescaling(1./255)
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

test_loss, test_acc = model.evaluate(test_dataset)
print(f"✅ Test Accuracy: {test_acc:.4f}")

def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224)) 
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class]} ({confidence * 100:.2f}%)")
    plt.axis('off')
    plt.show()

def get_random_image_path(dataset_path):
    class_dirs = [os.path.join(dataset_path, class_name) for class_name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, class_name))]
    random_class_dir = random.choice(class_dirs)
    random_image_path = os.path.join(random_class_dir, random.choice(os.listdir(random_class_dir)))
    return random_image_path

sample_image_path = get_random_image_path(dataset_path)
predict_image(sample_image_path)