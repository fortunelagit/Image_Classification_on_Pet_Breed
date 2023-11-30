import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from keras.utils import img_to_array
model = keras.models.load_model('./model.h5')

breed = ['Abyssinian', 'American Bulldog', 'American Pitbull Terrier', 'Basset Hound', 'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier']
dog_breed = [1, 2, 3, 4, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 25, 28, 29, 30, 31, 34, 35, 36]
cat_breed = [0, 5, 6, 7, 9, 11, 20, 23, 26, 27, 32, 33]

def transform_image(pillow_image):
    data = img_to_array(pillow_image)
    data = np.expand_dims(data, axis = 0)
    data = data / 255.
    data = tf.image.resize(data, [300, 300])
    
    return data


def predict(x):
    predictions = model(x)
    predictions = tf.nn.softmax(predictions)
    print("pass prdict")
    pred0 = predictions[0]
    label0 = np.argmax(pred0)
    return label0

def predict_all(x):
    predictions = model(x)
    predictions = tf.nn.softmax(predictions)
    class_probabilities = predictions.numpy()[0]  # Convert TensorFlow tensor to NumPy array
    return class_probabilities

def get_species(x):
    if x in cat_breed:
        return 'Cat'
    if x in dog_breed:
        return 'Dog'

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes))
            tensor = transform_image(pillow_img)
            class_probabilities = predict(tensor)

            # Assuming 'breed' is defined somewhere
            prediction = np.argmax(class_probabilities)
            species = ' (' + get_species(prediction) + ')'
            
            data = {"prediction": breed[prediction] + species, "class_probabilities": class_probabilities.tolist()}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"
# def index():
#     if request.method == "POST":
#         file = request.files.get('file')
#         if file is None or file.filename == "":
#             return jsonify({"error": "no file"})

#         try:
#             image_bytes = file.read()
#             pillow_img = Image.open(io.BytesIO(image_bytes))
#             tensor = transform_image(pillow_img)
#             prediction = predict(tensor)
            
#             species = ' (' + get_species(prediction) + ')'
#             data = {"prediction": breed[prediction] + species}
#             return jsonify(data)
#         except Exception as e:
#             return jsonify({"error": str(e)})

#     return "OK"


if __name__ == "__main__":
    app.run(debug=True)