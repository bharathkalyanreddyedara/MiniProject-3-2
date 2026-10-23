@echo off
echo ============================================
echo QUICK START - Model Training
echo ============================================
echo.

echo This will create a sample dataset and train models.
echo For production, use a real malicious URL dataset!
echo.
pause

call venv\Scripts\activate

echo.
echo [1/2] Creating sample dataset...
python create_sample_dataset.py

echo.
echo [2/2] Training models (this may take 5-10 minutes)...
python train_models.py

echo.
echo ============================================
echo DONE! You can now run the application.
echo ============================================
echo.
echo To start the server, run: run.bat
echo Or manually: python app.py
echo.
pause
