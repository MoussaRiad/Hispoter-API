import numpy as np
import cv2

from fastapi import APIRouter, File, UploadFile, Form

from io import BytesIO
from PIL import Image

from services.imageService import read_image


images = APIRouter()

@images.post("/normalize")
async def nomalize(image: UploadFile = File(...)):
    """
    Perform skew correction on an image using OpenCV.
    """
    img = await read_image(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # return await save_image(rotated)
    # convert to png file format of the image in binary form 
    png = Image.fromarray(rotated)
    image_buffer = BytesIO()
    png.save("normalized-"+".png")
    image_buffer.seek(0)

    # Return the binary representation of the image
    return image_buffer.getvalue()

@images.post("/normalize/rotate")
async def rotate_image(image: UploadFile = File(...), angle: float = Form(...)):
    """
    Rotate an image using OpenCV.
    """
    img = await read_image(image)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    png = Image.fromarray(rotated)
    # image_buffer = BytesIO()
    name="rotated-"+str(angle)+".png"
    # png.save(name)
    print(f"Image rotated: {name}")
    file = await save_image(rotated,name=name)
    # image_buffer.seek(0)
    # file= image_buffer.getvalue()
    return Response(file, media_type="image/png")

@images.post("/normalize/resize")
async def resize_image(image: UploadFile = File(...), width: int = Form(...), height: int = Form(...)):
    """
    Resize an image using OpenCV.
    """
    img = await read_image(image)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    png = Image.fromarray(resized)
    # image_buffer = BytesIO()
    png.save("resized-"+str(width)+"-"+str(height)+".png")
    # image_buffer.seek(0)
    # return image_buffer.getvalue()

@images.post("/normalize/contrast_stretching")
async def contrast_stretching(image: UploadFile = File(...), min_in: int = 0, max_in: int = 255, min_out: int = 0, max_out: int = 255):
    # Read the uploaded image
    image_data = await image.read()

    # Convert image data to PIL Image
    pil_image = Image.open(BytesIO(image_data)).convert('L')

    # Apply contrast stretching
    stretched_image = apply_contrast_stretching(pil_image, min_in, max_in, min_out, max_out)

    # Convert processed image back to bytes
    # output_buffer = BytesIO()
    # stretched_image.save(output_buffer, format='JPEG')
    # output_buffer.seek(0)
    # processed_image_data = output_buffer.getvalue()
    file = await save_image(stretched_image,name='stretched')
    return {"processed_image": file}

@images.post("/preprocess_image")
async def preprocess_image_api(image: UploadFile =Form(...),greyMethod: str=Form('average'),noiseMethod: str=Form('gaussian')):
    print(f"greyMethod: {greyMethod}, noiseMethod: {noiseMethod}")
    contents = await read_image(image)
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    binary = preprocess_image(img=img, noiseMethod=noiseMethod, greyMethod=greyMethod)
    # print(f"Image binarized, shape: {binary.shape}")
    return {"image": binary.__str__()}

@images.post("/binarize_image")
async def binarize_image_api(image: UploadFile =Form(...),method: str=Form('niblack'),k: float = Form(-0.2), window_size: int = Form(15) ):
    contents = await read_image(image)
    name='binarized-'+method+'-'+str(k)+'-'+str(window_size)
    print(f"method: {method}, k: {k}, window_size: {window_size}")
    # nparr = np.fromstring(contents, np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    binary = binarize_image(img=contents, method=method, k=k, window_size=window_size,name=name)
    # print(f"Image binarized, shape: {binary.shape}")
    file = await save_image(binary,name=name)
    # print(f"Image binarized: {file.filename}")
    return Response(file, media_type="image/png")
    # return {'message':'Image binarized successfully','content':await file.read(),'file_size':await get_file_size(file)}
