import tensorflow as tf
import numpy as np
import os
import cv2

mnist = tf.keras.datasets.fashion_mnist
(images_train, labels_train), (images_test, labels_test) = mnist.load_data()

images_train = images_train / 255.0
images_test = images_test / 255.0

def resize_and_convert(image, label):
    image = tf.image.resize(image[..., tf.newaxis], (224, 224))
    image = tf.image.grayscale_to_rgb(image)
    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
train_dataset = train_dataset.map(resize_and_convert).batch(32).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
test_dataset = test_dataset.map(resize_and_convert).batch(32).prefetch(tf.data.AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, epochs=10, validation_data=test_dataset)

model.save('model_mobilenetv2.keras')

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not loaded correctly. Please check the file path.")
    
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized / 255.0
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)
    
    return image_expanded

def predict_class(image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    return predicted_class

if __name__ == "__main__":
    image_path = input("Enter the path to the clothing apparel image: ")
    
    try:
        predicted_class = predict_class(image_path)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        print(f"Predicted class: {class_names[predicted_class]}")
        
    except Exception as e:
        print(f"Error: {e}")
