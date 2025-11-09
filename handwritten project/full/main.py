import joblib
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(int) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

clf = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=1000, C=1.0, n_jobs=-1, verbose=1)

start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()

joblib.dump(clf, "/Users/shahnawazaadil/Desktop/Github/ADVI/handwritten project/full/mnist_full_model.pkl")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Full MNIST Model Accuracy: {accuracy:.4f}")
print(f"Training Time: {end_time - start_time:.2f} seconds")

print("Model saved as 'mnist_full_model.pkl'")