from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import joblib

# inicializa o rate limiter
limiter = Limiter(key_func=get_remote_address)

# inicializa a API
app = FastAPI()

# conecta o handler do slowapi
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# carrega modelo (ajuste para seu modelo real)
model = joblib.load("modelo.pkl")

@app.post("/predict")
@limiter.limit("1/3seconds")  # 1 requisição a cada 3 segundos
async def predict(request: Request):
    data = await request.json()
    texto = data.get("text")

    if not texto:
        return {"error": "missing 'text' field"}

    prediction = model.predict([texto])
    return {"prediction": prediction[0]}
