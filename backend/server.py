from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import json
import io
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from openai import AsyncOpenAI
from urllib.parse import urlsplit, urlunsplit, quote_plus

# numpy is used indirectly, but we need it for type checks
import numpy as np

warnings.filterwarnings("ignore")


# =====================================================
# ENV / LOGS
# =====================================================
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("tsa-api")


def _get_env(name: str, required: bool = False, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if required and (val is None or str(val).strip() == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


# =====================================================
# JSON SAFE (IMPORTANT FIX FOR numpy types)
# =====================================================
def to_jsonable(obj: Any) -> Any:
    """
    Recursively convert numpy / pandas types into JSON-serializable Python types.
    Fixes: numpy.bool_, numpy.int64, numpy.float64, numpy arrays, pandas timestamps.
    """
    # primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # numpy scalar types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]

    # pandas types
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)

    # dict
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]

    # fallback: try to coerce
    try:
        if hasattr(obj, "item"):
            return to_jsonable(obj.item())
    except Exception:
        pass

    # last resort: string
    return str(obj)


# =====================================================
# MONGODB (SAFE URI)
# =====================================================
def sanitize_mongo_uri(uri: str) -> str:
    parts = urlsplit(uri)
    netloc = parts.netloc
    if "@" not in netloc:
        return uri

    userinfo, hostpart = netloc.rsplit("@", 1)
    if ":" in userinfo:
        user, pwd = userinfo.split(":", 1)
        new_netloc = f"{quote_plus(user)}:{quote_plus(pwd)}@{hostpart}"
    else:
        new_netloc = f"{quote_plus(userinfo)}@{hostpart}"

    return urlunsplit((parts.scheme, new_netloc, parts.path, parts.query, parts.fragment))


mongo_url_raw = _get_env("MONGO_URL", required=True)
mongo_url = sanitize_mongo_uri(mongo_url_raw)

client = AsyncIOMotorClient(
    mongo_url,
    serverSelectionTimeoutMS=8000,
    connectTimeoutMS=8000,
    socketTimeoutMS=8000,
)

db_name = _get_env("DB_NAME", required=True)
db = client[db_name]
logger.info("Mongo configured (db=%s)", db_name)


# =====================================================
# FASTAPI
# =====================================================
app = FastAPI()
api_router = APIRouter(prefix="/api")


# =====================================================
# CORS (IMPORTANT)
# =====================================================
# Render env example:
# CORS_ORIGINS=https://tsafr0.netlify.app,http://localhost:3000
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
origins = [o.strip().rstrip("/") for o in cors_origins.split(",") if o.strip()]

logger.info("CORS origins=%s | allow_credentials=False | regex=netlify", origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https://.*\.netlify\.app$",  # allow all netlify deploys
    allow_credentials=False,  # ✅ MUST be False here
    allow_methods=["*"],
    allow_headers=["*"],
)

# preflight safety
@app.options("/{path:path}")
async def options_handler(path: str):
    return JSONResponse(content={"ok": True})


# =====================================================
# MODELS
# =====================================================
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatusCheckCreate(BaseModel):
    client_name: str


class ColumnSelectionRequest(BaseModel):
    file_id: str
    date_column: str
    value_column: str
    duplicate_strategy: str = "mean"


class ForecastRequest(BaseModel):
    file_id: str
    date_column: str
    value_column: str
    duplicate_strategy: str = "mean"
    p: int
    d: int
    q: int
    seasonal: bool = False
    P: int = 0
    D: int = 0
    Q: int = 0
    m: int = 12
    forecast_horizon: int = 12


class AIReportRequest(BaseModel):
    file_id: str
    analysis_data: Dict[str, Any]
    model_type: str = "gpt-5-mini"
    report_mode: str = "court"


# =====================================================
# HELPERS
# =====================================================
def parse_dataframe(file_content: bytes, filename: str) -> pd.DataFrame:
    try:
        if filename.endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_content))
        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(file_content))
        raise ValueError("Format de fichier non supporté. Utilisez CSV ou Excel.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de parsing: {str(e)}")


def prepare_timeseries(df: pd.DataFrame, date_col: str, value_col: str, duplicate_strategy: str = "mean") -> pd.Series:
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, value_col])
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col]).sort_values(date_col)

        if duplicate_strategy == "mean":
            df = df.groupby(date_col)[value_col].mean().reset_index()
        elif duplicate_strategy == "sum":
            df = df.groupby(date_col)[value_col].sum().reset_index()

        df.set_index(date_col, inplace=True)
        return df[value_col]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de préparation des données: {str(e)}")


def perform_stl_decomposition(series: pd.Series, period: int = None) -> Dict[str, Any]:
    try:
        if period is None:
            inferred = pd.infer_freq(series.index)
            period = {"D": 7, "M": 12, "Q": 4, "Y": 1, "W": 52}.get(inferred[0], 12) if inferred else 12

        if len(series) < 2 * period:
            return {
                "success": False,
                "error": f"Série trop courte pour STL (>= {2*period} obs, obtenu {len(series)})",
                "period": int(period),
            }

        stl = STL(series, seasonal=period)
        result = stl.fit()

        return {
            "success": True,
            "trend": result.trend.tolist(),
            "seasonal": result.seasonal.tolist(),
            "resid": result.resid.tolist(),
            "dates": series.index.astype(str).tolist(),
            "period": int(period),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "period": int(period or 12)}


def perform_adf_test(series: pd.Series) -> Dict[str, Any]:
    try:
        adf_stat, p_value, usedlag, nobs, critical_values, _ = adfuller(series.dropna())
        is_stationary = p_value < 0.05  # can be numpy.bool_ depending on types

        return {
            "adf_statistic": float(adf_stat),
            "p_value": float(p_value),
            "used_lag": int(usedlag),
            "n_obs": int(nobs),
            "critical_values": {k: float(v) for k, v in critical_values.items()},
            "is_stationary": bool(is_stationary),  # ✅ ensure python bool
            "interpretation": (
                f"La série est {'stationnaire' if is_stationary else 'non-stationnaire'} "
                f"(p-value = {p_value:.4f}). "
                f"{'Pas besoin de différenciation.' if is_stationary else 'Différenciation recommandée.'}"
            ),
        }
    except Exception as e:
        return {"error": str(e), "interpretation": "Impossible d'effectuer le test ADF"}


def train_forecast_model(series: pd.Series, p: int, d: int, q: int, P: int = 0, D: int = 0, Q: int = 0, m: int = 12, forecast_horizon: int = 12) -> Dict[str, Any]:
    try:
        min_obs = max(p + d + P + D + m, 10)
        if len(series) < min_obs:
            return {"success": False, "error": f"Série trop courte (>= {min_obs} obs)"}

        seasonal = P > 0 or D > 0 or Q > 0
        seasonal_order = (P, D, Q, m) if seasonal else (0, 0, 0, 0)

        model = SARIMAX(series, order=(p, d, q), seasonal_order=seasonal_order)
        fitted = model.fit(disp=False)

        fc = fitted.get_forecast(steps=forecast_horizon)
        mean = fc.predicted_mean
        ci = fc.conf_int()

        freq = pd.infer_freq(series.index) or "D"
        forecast_index = pd.date_range(start=series.index[-1], periods=forecast_horizon + 1, freq=freq)[1:]

        return {
            "success": True,
            "forecast_values": mean.tolist(),
            "forecast_dates": forecast_index.astype(str).tolist(),
            "lower_ci": ci.iloc[:, 0].tolist(),
            "upper_ci": ci.iloc[:, 1].tolist(),
            "in_sample_pred": fitted.fittedvalues.tolist(),
            "in_sample_dates": series.index.astype(str).tolist(),
            "observed_values": series.tolist(),
            "residuals": fitted.resid.tolist(),
            "model_info": {
                "aic": float(fitted.aic),
                "bic": float(fitted.bic),
                "model_type": "SARIMA" if seasonal else "ARIMA",
                "order": f"({p},{d},{q})",
                "seasonal_order": f"({P},{D},{Q},{m})" if seasonal else "None",
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def generate_ai_report(analysis_data: Dict[str, Any], report_mode: str = "court", model_type: str = "gpt-4o-mini") -> str:
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "⚠️ OPENAI_API_KEY manquante sur le backend (Render)."

        prompt = f"""Vous êtes un expert en analyse de séries temporelles.
DONNÉES:
{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Mode: {report_mode}. Donnez un rapport {'détaillé' if report_mode=='long' else 'concis'} et actionnable."""

        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model=model_type,
            messages=[
                {"role": "system", "content": "Expert séries temporelles."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Erreur IA: {str(e)}"


# =====================================================
# ROUTES
# =====================================================
@api_router.get("/")
async def root():
    return {"message": "Hello World"}


@api_router.post("/timeseries/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".csv", ".xlsx", ".xls")):
            raise HTTPException(status_code=400, detail="Format non supporté. Utilisez CSV ou Excel.")

        content = await file.read()
        df = parse_dataframe(content, file.filename)

        file_id = str(uuid.uuid4())
        await db.timeseries_files.insert_one({
            "file_id": file_id,
            "filename": file.filename,
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "columns": df.columns.tolist(),
            "n_rows": len(df),
            "data": df.to_json(orient="split", date_format="iso"),
        })

        return to_jsonable({
            "file_id": file_id,
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "n_rows": len(df),
            "preview": df.head(5).to_dict(orient="records"),
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("upload_file error")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@api_router.post("/timeseries/analyze")
async def analyze_timeseries(request: ColumnSelectionRequest):
    try:
        doc = await db.timeseries_files.find_one({"file_id": request.file_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Fichier non trouvé")

        df = pd.read_json(io.StringIO(doc["data"]), orient="split")
        series = prepare_timeseries(df, request.date_column, request.value_column, request.duplicate_strategy)

        stl_result = perform_stl_decomposition(series)
        adf_result = perform_adf_test(series)

        summary = {
            "n_observations": int(len(series)),
            "start_date": str(series.index[0]),
            "end_date": str(series.index[-1]),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "missing_values": int(series.isna().sum()),
            "frequency": str(pd.infer_freq(series.index) or "Non détectée"),
        }

        payload = {
            "original": {"dates": series.index.astype(str).tolist(), "values": series.tolist()},
            "stl": stl_result,
            "adf": adf_result,
            "summary": summary,
        }

        # ✅ critical: convert everything to JSON-safe
        return to_jsonable(payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("analyze_timeseries error")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@api_router.post("/timeseries/forecast")
async def forecast_timeseries(request: ForecastRequest):
    try:
        doc = await db.timeseries_files.find_one({"file_id": request.file_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Fichier non trouvé")

        df = pd.read_json(io.StringIO(doc["data"]), orient="split")
        series = prepare_timeseries(df, request.date_column, request.value_column, request.duplicate_strategy)

        result = train_forecast_model(
            series,
            request.p, request.d, request.q,
            request.P, request.D, request.Q, request.m,
            request.forecast_horizon,
        )
        return to_jsonable(result)
    except Exception as e:
        logger.exception("forecast_timeseries error")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/timeseries/ai-report")
async def generate_report(request: AIReportRequest):
    try:
        report = await generate_ai_report(request.analysis_data, request.report_mode, request.model_type)
        return {"report": report}
    except Exception as e:
        logger.exception("ai-report error")
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(api_router)


@app.get("/")
async def main_root():
    return {"message": "TSA API - Time Series Analysis", "docs": "/docs"}
