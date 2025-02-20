from fastapi import FastAPI

app = FastAPI(title="Belajar Model Inference", version="1.0.1")

@app.get("/")
async def MainFile():
  """
  Main Router
  """
  return {"result":"Simple Model Inference on FastAPI"}

@app.post("/logits")
async def GetResult()