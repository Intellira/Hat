from fastapi import FastAPI

app = FastAPI(title="Belajar Model Inference", version="1.0.0")

@app.on_event("startup")
def load_clf():
  """
  Automatically Load Pickle Model
  """
  with open('model.pkl', 'rb') as rfile:

    global model
    
    model = pickle.load(rfile)

@app.get("/")
async def MainFile():
  """
  Main Router
  """
  return {"result":"Simple Model Inference on FastAPI"}

@app.post("/logits")
async def GetResult()