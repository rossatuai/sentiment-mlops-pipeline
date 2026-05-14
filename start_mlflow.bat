@echo off
CALL C:\Users\rossm\anaconda3\Scripts\activate.bat
CALL conda activate sentiment-mlops
python -m mlflow ui --port 5001
pause