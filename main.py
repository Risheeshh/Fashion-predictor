import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("model.keras")

mnist = tf.keras.datasets.fashion_mnist
(images_train, labels_train), (images_test, labels_test) = mnist.load_data()

images_train, images_test = images_train / 255.0, images_test / 255.0

image_path = input("Enter the image path")
image = cv2.imread(f'{image_path}', cv2.IMREAD_GRAYSCALE)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows
image = cv2.resize(image, (28,28))

if image is None:
    print("Error: Unable to load image.")
else:
    image_array = np.invert(np.array([image]))
    print("Shape of the image array:", image_array.shape)

img = image_array / 255.0
for i in range(0, 27, 1):
    for j in range(0, 27, 1):
        img[i][j] = 1 - img[i][j]

img = np.expand_dims(img, axis=0)
print(img)
img = img.astype(np.uint8)

predictions = model.predict(img)
predicted_class = np.argmax(predictions, axis=1)

print("Predicted label:", predicted_class)
