from fastapi import FastAPI
import uvicorn

app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

@app.get("/")
def root():
    return {"message": "Backend Working"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/test")
def test():
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080
    )