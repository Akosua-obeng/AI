import glob
import os
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

...
# Specify the directory where your images are
image_dir = r"C:\Users\amobe\Pictures"
image_exts = ["jpg", "jpeg", "png"]  # specify the image file extensions to search for

# Open a file to write the captions
with open("captions.txt", "w") as caption_file:
# Iterate over each image file in the directory
    for image_ext in image_exts:
        for img_path in glob.glob(os.path.join(image_dir, f"*.{image_ext}")):
            # Load your image
            raw_image = Image.open(img_path).convert('RGB')
            