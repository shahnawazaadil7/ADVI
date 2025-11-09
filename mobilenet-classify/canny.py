import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def download_image(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    else:
        print("Failed to download image")
        return None

coco_image_url = "http://images.cocodataset.org/train2017/000000581929.jpg"

image = download_image(coco_image_url)
if image is not None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
    edges = cv2.Canny(blurred, 50, 150) 

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edge Detection")
    plt.axis("off")

    plt.show()