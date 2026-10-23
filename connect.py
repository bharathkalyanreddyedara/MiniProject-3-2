import pytesseract
from PIL import Image

# Specify path (IMPORTANT for Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = Image.open("/uploads/IMG-20250128-WA02241.jpg")
text = pytesseract.image_to_string(img)

print(text)