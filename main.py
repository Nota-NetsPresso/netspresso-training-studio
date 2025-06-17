import uvicorn
from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from src.api.api import api_router
from src.configs.settings import settings
from src.configs.version import BACKEND_VERSION
from src.core.db.session import init_db
from src.core.middlewares.context_middleware import ContextMiddleware


def init_routers(app: FastAPI) -> None:
    app.include_router(api_router, prefix=settings.API_PREFIX)


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
    # Initialize database
    init_db()

    app = FastAPI(
        title="NetsPresso Training Studio",
        version=BACKEND_VERSION,
        middleware=make_middleware(),
    )
    init_routers(app=app)
    app.add_middleware(ContextMiddleware)

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
