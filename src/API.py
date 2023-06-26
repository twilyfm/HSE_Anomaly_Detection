import pickle
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Request
from pydantic import BaseModel
from typing import List
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
import logging
from flask import request
from prometheus_client import Counter, start_http_server
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator

from flask import render_template
import shutil
import os
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup prometheus metrics
model_loads_counter = Counter('model_loads', 'Number of times the model is loaded')
uploads_counter = Counter('uploads', 'Number of times a file is uploaded')
predictions_counter = Counter('predictions', 'Number of times a prediction is made')

# Загрузка модели
async def load_model(folder="model", name="catboost_classifier.pkl"):
    try:
        file_path = os.path.join(folder, name)
        model = pickle.load(open(file_path, "rb"))
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error("An error occurred while loading the model: %s", str(e))

# Create a FastAPI instance
app = FastAPI()
templates = Jinja2Templates(directory="html")
UPLOAD_FOLDER = 'uploaded'

# Создание модели для запроса
class TransactionData(BaseModel):
    features: list[float]

async def to_api_view(list_:List):
    preds = dict()
    for i, item in enumerate(list_):
        name = f"prediction_{i}"
        preds[name] = item
    return preds



@app.get("/")
async def root(request: Request):
    logger.info("Received a request to the root endpoint")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/doc/", response_class=HTMLResponse)
async def doc(request: Request):
    return templates.TemplateResponse("doc.html", {"request": request})

@app.get("/upload/", response_class=HTMLResponse)
async def upload(request: Request):
   return templates.TemplateResponse("uploadfile.html", {"request": request})

ALLOWED_EXTENSIONS = {'csv', 'parquet'}

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    uploads_counter.inc()
    try:
        filename = file.filename
        extension = filename.split('.')[-1]
        if extension.lower() not in ALLOWED_EXTENSIONS:
            return {"error": "Invalid file format. Only CSV and Parquet files are allowed."}

        file_location = f"files/{filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        logging.info(f"File {filename} has been successfully uploaded.")
        return {"info": f"File '{filename}' saved at {file_location}"}
    except Exception as e:
        logging.error(f"An error occurred while uploading the file: {e}")
        return {"error": "An error occurred while uploading the file."}

@app.post("/uploader/", response_class=HTMLResponse)
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    uploads_counter.inc()
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return templates.TemplateResponse("uploaded.html", {"request": request, "filename": file.filename})

@app.get("/inference_file", response_class=HTMLResponse)
async def upload(request: Request):
    files = os.listdir(UPLOAD_FOLDER)
    return templates.TemplateResponse("choose_file.html", {"request": request, "files": files})

@app.post("/predict")
async def predict_fraud(infer_file: str = Form(...)):
    predictions_counter.inc()
    file_path = os.path.join(UPLOAD_FOLDER, infer_file)
    if '.parquet' in file_path:
        input_data = pd.read_parquet("merge_filled_without_drop.parquet", engine='pyarrow').sample(n=100)
    elif '.csv' in file_path:
        input_data = pd.read_csv(file_path)
    model = await load_model()
    predictions = model.predict(input_data.drop("isFraud", axis=1))
    input_data['predictions'] = predictions
    output_file_path = os.path.join(UPLOAD_FOLDER, "output.csv")
    input_data.to_csv(output_file_path, index=False)
    return FileResponse(output_file_path, media_type='application/csv', filename="output.csv")

@app.get("/files", response_class=HTMLResponse)
async def files(request: Request):
    files = os.listdir(UPLOAD_FOLDER)
    return templates.TemplateResponse("file_list.html", {"request": request, "files": files})

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
            return {"message": f"The file {filename} has been deleted."}
        except Exception as e:
            return {"error": f"Error occurred: {e}"}
    else:
        return {"error": f"The file {filename} does not exist."}


if __name__ == '__main__':
    start_http_server(8000)
    uvicorn.run(app, host='127.0.0.1', port=8001)


