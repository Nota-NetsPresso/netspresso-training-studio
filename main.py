import uvicorn
from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from src.configs.settings import settings


def init_routers(app: FastAPI) -> None:
    # app.include_router(api_router)
    pass


def make_middleware() -> list[Middleware]:
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["Access-Token", "Authorization", "Content-Deposition"],
        ),
    ]

    return middleware


def create_app():
    app = FastAPI(
        title="NetsPresso Training Studio",
        version="0.1.0",
        middleware=make_middleware(),
    )
    init_routers(app=app)

    return app


app = create_app()

@app.get("/")
def healthcheck():
    return {"message": "Hello NetsPresso Training Studio!"}


if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=settings.SERVER_PORT,
        workers=settings.SERVER_WORKERS,
    )
