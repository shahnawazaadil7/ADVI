import joblib
import matplotlib.pyplot as plt

clf = joblib.load("/Users/shahnawazaadil/Desktop/Github/ADVI/handwritten project/full/mnist_full_model.pkl")

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(int)

sample_images = X[:5]
sample_labels = y[:5]

predictions = clf.predict(sample_images)

plt.figure(figsize=(10, 4))
for i, (image, label) in enumerate(zip(sample_images, sample_labels)):
    plt.subplot(1, 5, i + 1)
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.title(f"Actual: {label}\nPred: {predictions[i]}")
    plt.axis("off")
plt.show()