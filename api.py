from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from main import preprocess_and_predict

app = FastAPI()


class NewsInput(BaseModel):
    text: str


# Load your trained model
fake_news_detection_model = joblib.load('model.sav')


@app.post("/predict")
async def predict(news: NewsInput):
    # Preprocess and predict the news
    prediction = preprocess_and_predict(news.text)
    return {"prediction": prediction}
