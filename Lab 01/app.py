import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Belajar Model Inference")

@app.on_event('startup')
def ModelOnCall():
  url = 'iris_classifier.pkl'
  with open(url, 'rb') as ModelFile:
    global model
    model = pickle.load(ModelFile)

labels = ['Setosa', 'Versicolor', 'Virginica']

class IrisClass(BaseModel):
  sepal_length : float
  sepal_width : float
  petal_length : float
  petal_width : float

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
  irises = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
  logits = model.predict(irises).tolist()
  result = labels[logits[0]]
  return {"result" : result}