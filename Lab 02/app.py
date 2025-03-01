import pickle
import numpy as np
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel, conlist

app = FastAPI(title="Belajar Model Inference Batches", version="1.0.1")

@app.on_event('startup')
def ModelOnCall():
  url = '../Lab 01/iris_classifier.pkl'
  with open(url, 'rb') as ModelFile:
    global model
    model = pickle.load(ModelFile)

labels = ['Setosa', 'Versicolor', 'Virginica']

class IrisClass(BaseModel):
  """
  Iris Class Base Model
  """
  batches : List[conlist(item_type=float, min_length=4, max_length=4)]

@app.get("/")
async def MainRoute():
  """
  Main Route
  """
  return {"result" : "Simple Model Inference Batches"}

@app.post('/logits')
async def GetResult(iris : IrisClass):
  """
  Get Model Result
  """
  batches = np.array(iris.batches)
  logits = model.predict(batches).tolist()
  result = [labels[i] for i in logits]
  return {"result" : result}