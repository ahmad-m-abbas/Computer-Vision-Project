import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Function to convert image to pixel array
def image_to_pixel_array(filepath):
    image = Image.open(filepath)
    return np.array(image).flatten()

# Lists for pixels and labels
pixels_list = []
labels_list = []

# List of Arabic characters and fonts
arabic_characters = ['أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
font_paths = ["18 Khebrat Musamim Regular.ttf", "AAAGoldenLotus Stg1_Ver1 Regular.ttf", "(A) Arslan Wessam A (A) Arslan Wessam A.ttf", "KFGQPC Uthman Taha Naskh Regular.ttf"]  # Add paths for 10 different fonts

# Directory to save images
output_dir = "arabic_characters_images"
os.makedirs(output_dir, exist_ok=True)

# Image size
img_size = (32, 32)

label = 1
for char in arabic_characters:
    for font_path in font_paths:
        img = Image.new('L', img_size, color = 0)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(font_path, 28)
            w, h = draw.textsize(char, font=font)
            draw.text(((img_size[0]-w)/2, (img_size[1]-h)/2), char, font=font, fill=255)
            image_path = os.path.join(output_dir, f'{label}_{char}_{os.path.basename(font_path)}.png')
            img.save(image_path)
            pixels_list.append(image_to_pixel_array(image_path))
            labels_list.append(label)
        except Exception as e:
            print(f"An error occurred with font {font_path}: {e}")
    label += 1

# Save pixel data to CSV
with open('pixel_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(pixels_list)


with open('labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for label in labels_list:
        writer.writerow([label])

print("CSV files generated successfully.")


