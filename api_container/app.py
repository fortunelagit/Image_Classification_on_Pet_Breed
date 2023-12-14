import os
import io
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, templating
import json

# Use TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='./model.tflite')
interpreter.allocate_tensors()

breed = ['Abyssinian', 'American Bulldog', 'American Pitbull Terrier', 'Basset Hound', 'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier']
dog_breed = [1, 2, 3, 4, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 25, 28, 29, 30, 31, 34, 35, 36]
cat_breed = [0, 5, 6, 7, 9, 11, 20, 23, 26, 27, 32, 33]

def transform_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize((300, 300))
    data = np.array(img, dtype=np.float32)  # Explicitly set the data type to FLOAT32
    data = np.expand_dims(data, axis=0)
    data = data / 255.

    return data



def predict_top3_formatted(x):
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], x)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    predictions = tf.nn.softmax(output)

    pred0 = predictions[0]
    label0 = np.argmax(pred0)

    class_probabilities = predictions.numpy()[0]

    # Get indices of the top 3 probabilities for each class
    top3_indices_per_class = np.argsort(class_probabilities)[::-1][:3]

    # Get the corresponding probabilities and classes
    top3_probabilities_per_class = class_probabilities[top3_indices_per_class]
    top3_classes = top3_indices_per_class


    # Format the results
    formatted_results = []
    for i in range(len(top3_classes)):
        breed_index = top3_classes[i]
        breed_name = breed[breed_index]
        probability = top3_probabilities_per_class[i] * 100  # Convert to percentage
        formatted_string = f"{breed_name} ({probability:.2f}%)"
        formatted_results.append(formatted_string)

    return label0, formatted_results


def get_species(x):
    if x in cat_breed:
        return 'Cat'
    if x in dog_breed:
        return 'Dog'

import json
from PIL import Image

import torch
from torchvision import transforms

from pytorch_pretrained_vit import ViT

model_name = 'B_16_imagenet1k'
model = ViT(model_name, pretrained=True)

# Load class names
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]


def transform_image_torch(img):
    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
    img = tfms(img).unsqueeze(0)

    return img

def classify_image_torch(img):
    # Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img).squeeze(0)
    top3_predictions = []

    # Get indices of the top 3 probabilities
    top3_indices = torch.topk(outputs, k=1).indices.tolist()

    for idx in top3_indices:
        prob = torch.softmax(outputs, -1)[idx].item()
        formatted_string = '{label} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob * 100)
        top3_predictions.append(formatted_string)

    return top3_predictions

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home():
    return templating.render_template('./index.html')

@app.route("/breed-api", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"messages": "no file received"})

        try:
            image_bytes = file.read()
            image_path = io.BytesIO(image_bytes)
            tensor = transform_image(image_path)
            prediction, top3values = predict_top3_formatted(tensor)

            img = Image.open(file)
            img = transform_image_torch(img)
            torch_data = classify_image_torch(img)

            
            species = ' (' + get_species(prediction) + ')'
            data = {"prediction": breed[prediction] + species, "values": top3values, "torch_values": torch_data, "messages": "prediction success"}
            json_data = json.dumps(data)

            return json_data
        except Exception as e:
            return jsonify({"error": str(e)})

    return jsonify({"messages": "API is running"})


if __name__ == "__main__":
    app.run(debug=False)
