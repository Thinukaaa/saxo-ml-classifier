# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .predict_model import predict_new_audio  # Use relative import if predict_model.py is in the same directory

app = FastAPI()

# Allow frontend requests (adjust origin in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "SaxoML API is live"}

@app.post("/predict")
def predict():
    result = predict_new_audio()
    return {"status": "success", "predictions": result}
