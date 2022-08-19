import fastapi
import predict_onnx

app = fastapi.FastAPI()


@app.post("/predict")
def predict(text: str):
    prediction = predict_onnx.main(text)
    return {"prediction": prediction}
