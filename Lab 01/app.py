import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Belajar Model Inference", version="1.0.0")

class IrisClass(BaseModel):
  """
  Iris Dataset Class
  """
  petal: float
  petal: float
  petal: float
  petal: float

@app.on_event("startup")
def load_clf():
  """
  Automatically Load Pickle Model
  """
  with open('model.pkl', 'rb') as file:
    global model
    model = pickle.load(file)

@app.get("/")
async def MainFile():
  """
  Main Router
  """
  return {"result":"Simple Model Inference on FastAPI"}

@app.post("/logits")
async def GetResult()