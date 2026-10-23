@echo off
echo ============================================
echo Installing OCR and QR Code Dependencies
echo ============================================
echo.

call venv\Scripts\activate

echo Installing Python packages...
pip install pytesseract pillow pyzbar

echo.
echo ============================================
echo Python packages installed!
echo ============================================
echo.
echo IMPORTANT: You still need to install Tesseract OCR
echo.
echo Download from:
echo https://github.com/UB-Mannheim/tesseract/wiki
echo.
echo After installing Tesseract, restart the server.
echo.
pause
