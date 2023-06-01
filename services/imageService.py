

from io import BytesIO
import cv2
from fastapi import APIRouter
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from skimage import img_as_ubyte



def apply_contrast_stretching(image, min_in, max_in, min_out, max_out):
    # Convert PIL Image to numpy array
    image_array = np.array(image)

    # Normalize pixel intensities within the desired input range
    normalized_image = np.clip(image_array, min_in, max_in)
    normalized_image = (normalized_image - min_in) * (max_out - min_out) / (max_in - min_in) + min_out

    # Convert processed image back to PIL Image
    stretched_image = Image.fromarray(normalized_image.astype(np.uint8))
    stretched_image.save("stretched_image.jpg")
    return np.array(stretched_image)

def normalize_image(img):
    # Skew correction
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # convert to png file format of the image in binary form 
    png = Image.fromarray(img)
    png.save("normalize"+''+".png")
    # return the png file format of the image in binary form
    return np.array(png)

def preprocess_image(img,greyMethod='average',noiseMethod='gaussian'):
    # Convert to grayscale
    if(greyMethod =='average') :img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif (greyMethod == 'luminosity'):img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:, :, 0]
    elif (greyMethod == 'desaturation'):img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    img = cv2.bitwise_not(img)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # Noise removal
    if noiseMethod == 'gaussian':
        img = cv2.GaussianBlur(img, (5, 5), 0)
    elif noiseMethod == 'median':
        img = cv2.medianBlur(img, 5)
    elif noiseMethod == 'bilateral':
        img = cv2.bilateralFilter(img, 9, 75, 75)

    # convert to png file format of the image in binary form 
    png = Image.fromarray(img)
    png.save("preprocess-"+greyMethod+'-'+noiseMethod+".png")
    # return the png file format of the image in binary form
    return np.array(png)

def binarize_image(img,k,window_size, method:str='otsu',name:str = 'binary'):
    """
    Binarize an image using OpenCV.
    Args:
        img: The image to binarize.
        method: The binarization method to use. Can be 'otsu', 'niblack', or 'sauvola'.
        k: The k value for niblack binarization.
        window_size: The window size for niblack and sauvola binarization.
    
    Returns:
        The binarized image.
    """

    # Binarization
    if method == 'otsu':
        thresh = threshold_otsu(img)
    elif method == 'niblack':
        thresh = threshold_niblack(img, window_size=window_size, k=k)
    elif method == 'sauvola':
        thresh = threshold_sauvola(img, window_size=window_size)
    else:
        return None
    binary = img_as_ubyte(img > thresh)
    print('image binarized with '+method+' method successfully'+str(binary.shape))
    # convert to png file format of the image in binary form 
    png = Image.fromarray(binary)
    # png.save(name+".png")
    
    # return the png file format of the image in binary form
    return np.array(png) 


async def read_image(file) -> np.ndarray:
    """
    Reads an image from an UploadFile object and returns it as a numpy array.
    """
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    return img


async def save_image(image: np.ndarray,name:str='output') :# UploadFile:
    """
    Saves an image as an UploadFile object and returns it.
    """
    pil_img = Image.fromarray(image)
    with BytesIO() as output:
        pil_img.save(output, format="PNG")
        # save_path = os.path.join("http://localhost:3000/", name + ".png")
        # pil_img.save(name+'.png',format="PNG")
        contents = output.getvalue()
    # return StreamingResponse(output, media_type="image/png")
    return contents
    