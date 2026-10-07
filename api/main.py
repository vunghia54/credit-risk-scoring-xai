"""FastAPI lifespan and HTTP endpoints for the frozen local inference service."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model_service import DEFAULT_ARTIFACT_PATH, DEFAULT_METADATA_PATH, ModelService
from api.schemas import DISCLAIMER, HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


def create_app(artifact_path: Path = DEFAULT_ARTIFACT_PATH,
               metadata_path: Path = DEFAULT_METADATA_PATH) -> FastAPI:
    """Construct the app without loading; each worker loads once in its lifespan."""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.model_service = ModelService.load(artifact_path, metadata_path)
        try:
            yield
        finally:
            application.state.model_service = None

    application = FastAPI(title="Credit Risk Scoring API", version="1.0.0",
                          description=DISCLAIMER + " Scores are uncalibrated model outputs; no lending decision is returned.",
                          lifespan=lifespan)

    def service(request: Request) -> ModelService:
        loaded = getattr(request.app.state, "model_service", None)
        if loaded is None:
            raise HTTPException(status_code=503, detail="Model is not available.")
        return loaded

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Keep FastAPI's 422 detail structure, omitting payload/context. This also
        # handles NaN/Infinity errors without emitting invalid JSON error bodies.
        details = [{"type": error["type"], "loc": list(error["loc"]), "msg": error["msg"]}
                   for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @application.get("/health", response_model=HealthResponse, summary="Check loaded model availability")
    def health(request: Request) -> HealthResponse:
        service(request)
        return HealthResponse(status="ok", model_loaded=True)

    @application.get("/model-info", response_model=ModelInfoResponse, summary="Inspect the frozen public model contract")
    def model_info(request: Request) -> ModelInfoResponse:
        return service(request).info

    @application.post("/predict", response_model=PredictionResponse, summary="Score one raw observation without a lending decision")
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        loaded = service(request)
        try:
            return loaded.predict(payload)
        except Exception as exc:
            # Do not log applicant values, exception messages or response stack traces.
            logger.error("Model inference failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Model inference failed.") from None

    return application


app = create_app()
