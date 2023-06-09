

import numpy as np
import cv2

from fastapi import APIRouter, File, UploadFile, Form

from io import BytesIO
from PIL import Image

from services.imageService import read_image

# Load the model
model = None#load_model('your_model_path')

# Load the word list
# with open('your_word_list_path', 'r') as f:
word_list =None#f.read().splitlines()

model = APIRouter()

@model.post('/predict')
async def predict(image: UploadFile = File(...)):
    # Read the uploaded image
    image_data = await image.read()

    # Convert image data to NumPy array
    nparr = np.frombuffer(image_data, np.uint8)

    # Decode the array as an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Reshape the image for the model input
    input_img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model.predict(input_img)

    # Get the top predicted classes
    top_classes = np.argsort(predictions)[0, ::-1][:10]  # Change the number 10 to the desired number of predictions

    # Get the corresponding words
    predicted_words = [word_list[class_idx] for class_idx in top_classes]

    # Return the ranked list of words as a response
    return {"predictions": predicted_words}

@model.post('/fine-tuning')
async def fine_tuning(image: UploadFile = File(...), label: str = Form(...)):
    # Read the uploaded image
    image_data = await image.read()

    # Convert image data to NumPy array
    nparr = np.frombuffer(image_data, np.uint8)

    # Decode the array as an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Reshape the image for the model input
    input_img = np.expand_dims(img, axis=0)

    # Get the one-hot encoded label
    label_one_hot = np.zeros((1, len(word_list)))
    label_index = word_list.index(label)
    label_one_hot[0, label_index] = 1

    # Train the model on the image
    model.train_on_batch(input_img, label_one_hot)

    # Return a success response
    return {"message": "Model fine-tuned successfully"}

# @model.post('/train')

@model.post('/models/code')
async def get_model_code():
    # Get the model code
    code = None
    return {"code": code}

