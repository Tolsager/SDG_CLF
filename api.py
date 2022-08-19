import os

import fastapi
import psutil
import pydantic

import predict_onnx
from sdg_clf import modelling, utils

os.environ["OMP_NUM_THREADS"] = f"{psutil.cpu_count()}"
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

app = fastapi.FastAPI()

onnx_model = modelling.create_model_for_provider("finetuned_models/roberta-large_1608124504.onnx")
tokenizer = utils.get_tokenizer("roberta-large")


class SDGText(pydantic.BaseModel):
    text: str


@app.post("/predict")
def predict(SDGText: SDGText):
    prediction = predict_onnx.main(SDGText.text, onnx_model, tokenizer)
    return {"prediction": prediction}


@app.get("/")
def index():
    return "Server is running"
