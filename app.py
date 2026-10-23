from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from pyzbar import pyzbar

# Configure Tesseract path for Windows (adjust if needed)
# Uncomment and modify if Tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ===============================================
# EXTRACTION CLASSES
# ===============================================

class StegoExtractor:
    def extract_lsb_text(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Invalid image")
        
        binary_data = ""
        for row in image:
            for pixel in row:
                b, g, r = pixel  # OpenCV uses BGR
                binary_data += str(r & 1)
                binary_data += str(g & 1)
                binary_data += str(b & 1)
        
        decoded_text = ""
        for i in range(0, len(binary_data), 8):
            byte = binary_data[i:i+8]
            if len(byte) < 8:
                break
            char_code = int(byte, 2)
            if char_code == 0:
                break
            if char_code < 32 or char_code > 126:
                if len(decoded_text) > 5:
                    break
            decoded_text += chr(char_code)
            if len(decoded_text) > 500:
                break
        
        return decoded_text

class OCRExtractor:
    """Extract visible text from images using OCR"""
    def extract_text(self, image_path):
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

class QRCodeExtractor:
    """Extract text from QR codes and barcodes in images"""
    def extract_qr_codes(self, image_path):
        try:
            image = cv2.imread(image_path)
            qr_codes = pyzbar.decode(image)
            
            results = []
            for qr in qr_codes:
                qr_data = qr.data.decode('utf-8')
                qr_type = qr.type
                results.append({
                    'type': qr_type,
                    'data': qr_data
                })
            
            return results
        except Exception as e:
            print(f"QR Code Error: {e}")
            return []

class URLValidator:
    def extract_urls(self, text):
        url_pattern = r'(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}[^\s]*)'
        urls = re.findall(url_pattern, text)
        cleaned_urls = []
        for url in urls:
            url = re.sub(r'[.,;:!?\)\]]+$', '', url)
            cleaned_urls.append(url)
        return cleaned_urls

class URLPreprocessor:
    def clean_url(self, url):
        url = url.lower()
        url = re.sub(r'https?://', '', url)
        url = re.sub(r'www\.', '', url)
        url = re.sub(r'[^a-z0-9./\-@]', '', url)
        return url

class HybridURLDetector:
    def __init__(self):
        self.bilstm_model = load_model("bilstm_url_model.h5")
        with open("tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open("rf_model.pkl", "rb") as f:
            self.rf_model = pickle.load(f)
        with open("hybrid_weight.pkl", "rb") as f:
            self.best_weight = pickle.load(f)
    
    def predict(self, url):
        seq = self.tokenizer.texts_to_sequences([url])
        pad_seq = pad_sequences(seq, maxlen=120)
        bilstm_prob = self.bilstm_model.predict(pad_seq, verbose=0)[0][0]
        
        tfidf = self.vectorizer.transform([url])
        rf_prob = self.rf_model.predict_proba(tfidf)[0][1]
        
        final_prob = (self.best_weight * bilstm_prob) + ((1-self.best_weight) * rf_prob)
        prediction = "MALICIOUS" if final_prob > 0.5 else "BENIGN"
        return prediction, float(final_prob)

# Initialize extractors
stego_extractor = StegoExtractor()
ocr_extractor = OCRExtractor()
qr_extractor = QRCodeExtractor()
validator = URLValidator()
preprocessor = URLPreprocessor()

# Check if model files exist
model_files = [
    "bilstm_url_model.h5",
    "tokenizer.pkl",
    "tfidf_vectorizer.pkl",
    "rf_model.pkl",
    "hybrid_weight.pkl"
]

missing_files = [f for f in model_files if not os.path.exists(f)]

if missing_files:
    print("\n⚠️  WARNING: Missing model files:")
    for f in missing_files:
        print(f"   - {f}")
    print("\nThe server will start, but analysis will fail without these files.\n")
    detector = None
else:
    detector = HybridURLDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if detector is None:
        return jsonify({'error': 'Model files not found. Please train the models first.'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. Extract steganographic hidden text
            stego_text = ""
            try:
                stego_text = stego_extractor.extract_lsb_text(filepath)
            except Exception as e:
                print(f"Steganography extraction error: {e}")
            
            # 2. Extract visible text using OCR
            ocr_text = ocr_extractor.extract_text(filepath)
            
            # 3. Extract QR codes
            qr_codes = qr_extractor.extract_qr_codes(filepath)
            
            # Combine all text sources
            all_text = f"{stego_text}\n{ocr_text}"
            for qr in qr_codes:
                all_text += f"\n{qr['data']}"
            
            # Extract URLs from all sources
            urls = validator.extract_urls(all_text)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_urls = []
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            # Analyze each URL
            results = []
            for url in unique_urls:
                clean_url = preprocessor.clean_url(url)
                prediction, prob = detector.predict(clean_url)
                
                # Determine source
                source = []
                if url in stego_text:
                    source.append('Steganography')
                if url in ocr_text:
                    source.append('OCR')
                for qr in qr_codes:
                    if url in qr['data']:
                        source.append(f"QR Code ({qr['type']})")
                
                results.append({
                    'url': url,
                    'prediction': prediction,
                    'probability': round(prob, 4),
                    'source': ', '.join(source) if source else 'Unknown'
                })
            
            os.remove(filepath)
            
            return jsonify({
                'stego_text': stego_text if stego_text else 'No steganographic data found',
                'ocr_text': ocr_text if ocr_text else 'No visible text detected',
                'qr_codes': qr_codes,
                'qr_count': len(qr_codes),
                'urls_found': len(unique_urls),
                'results': results
            })
        
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
