from fastapi import FastAPI, File, UploadFile,Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from starlette.requests import Request
from starlette.responses import Response
from starlette.templating import Jinja2Templates

from PIL import Image
from io import BytesIO
import cv2
import numpy as np

from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from skimage import img_as_ubyte

import tensorflow as tf
from keras.models import load_model

from services import preprocess_image, normalize_image, apply_contrast_stretching, save_image, read_image, binarize_image ,get_file_size


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cross-Origin Resource Sharing (CORS)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:3000/spotting",
]

app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
# Load the model
model = None#load_model('your_model_path')

# Load the word list
# with open('your_word_list_path', 'r') as f:
word_list =None#f.read().splitlines()

# Routes

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post('/predict')
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

