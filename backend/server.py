from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
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
import numpy as np
import json
import io
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from openai import AsyncOpenAI
from urllib.parse import urlsplit, urlunsplit, quote_plus

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
# MONGODB (SAFE URI)
# =====================================================
def sanitize_mongo_uri(uri: str) -> str:
    """
    Ensure username/password in MongoDB URI are RFC3986-escaped.
    Fixes passwords containing special chars like @ : / ? # %
    Works for mongodb:// and mongodb+srv://
    """
    parts = urlsplit(uri)
    netloc = parts.netloc  # can contain userinfo@host

    if "@" not in netloc:
        return uri  # no userinfo, nothing to escape

    userinfo, hostpart = netloc.rsplit("@", 1)

    # userinfo can be "user:pass" or just "user"
    if ":" in userinfo:
        user, pwd = userinfo.split(":", 1)
        user_enc = quote_plus(user)
        pwd_enc = quote_plus(pwd)
        new_netloc = f"{user_enc}:{pwd_enc}@{hostpart}"
    else:
        user_enc = quote_plus(userinfo)
        new_netloc = f"{user_enc}@{hostpart}"

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
# FASTAPI APP / ROUTER
# =====================================================
app = FastAPI(title="TSA API - Time Series Analysis")

api_router = APIRouter(prefix="/api")

# =====================================================
# CORS (FIXED)
# IMPORTANT:
# - Do NOT use allow_origins=["*"] with allow_credentials=True
# - For your React app, you don't need credentials -> keep False
# =====================================================
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,https://tsafr.netlify.app")
origins = [o.strip() for o in cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # ✅ explicit origins
    allow_credentials=False,    # ✅ important for "*" issues + you don't need cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS origins=%s", origins)

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
            df = pd.read_csv(io.BytesIO(file_content))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            raise ValueError("Format de fichier non supporté. Utilisez CSV ou Excel.")
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de parsing: {str(e)}")


def prepare_timeseries(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    duplicate_strategy: str = "mean"
) -> pd.Series:
    try:
        if date_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Colonne date introuvable: {date_col}")
        if value_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Colonne valeur introuvable: {value_col}")

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)

        if duplicate_strategy == "mean":
            df = df.groupby(date_col)[value_col].mean().reset_index()
        elif duplicate_strategy == "sum":
            df = df.groupby(date_col)[value_col].sum().reset_index()
        else:
            raise HTTPException(status_code=400, detail="duplicate_strategy doit être 'mean' ou 'sum'")

        df.set_index(date_col, inplace=True)
        return df[value_col]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de préparation des données: {str(e)}")


def perform_stl_decomposition(series: pd.Series, period: Optional[int] = None) -> Dict[str, Any]:
    try:
        if period is None:
            inferred_freq = pd.infer_freq(series.index)
            if inferred_freq:
                period = {"D": 7, "M": 12, "Q": 4, "Y": 1, "W": 52}.get(inferred_freq[0], 12)
            else:
                period = 12

        if len(series) < 2 * period:
            return {
                "success": False,
                "error": f"Série trop courte pour STL (>= {2 * period} obs), reçu {len(series)}",
                "period": period,
            }

        stl = STL(series, seasonal=period)
        result = stl.fit()

        return {
            "success": True,
            "trend": result.trend.tolist(),
            "seasonal": result.seasonal.tolist(),
            "resid": result.resid.tolist(),
            "dates": series.index.astype(str).tolist(),
            "period": period,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "period": period or 12}


def perform_adf_test(series: pd.Series) -> Dict[str, Any]:
    try:
        s = series.dropna()
        if len(s) < 5:
            return {"error": "Série trop courte pour ADF", "interpretation": "Impossible d'effectuer le test ADF"}

        result = adfuller(s)
        adf_stat, p_value, usedlag, nobs, critical_values, icbest = result

        is_stationary = p_value < 0.05
        interpretation = (
            f"La série est {'stationnaire' if is_stationary else 'non-stationnaire'} "
            f"(p-value = {p_value:.4f}). "
            f"{'Pas besoin de différenciation.' if is_stationary else 'Différenciation recommandée.'}"
        )

        return {
            "adf_statistic": float(adf_stat),
            "p_value": float(p_value),
            "used_lag": int(usedlag),
            "n_obs": int(nobs),
            "critical_values": {k: float(v) for k, v in critical_values.items()},
            "is_stationary": bool(is_stationary),
            "interpretation": interpretation,
        }
    except Exception as e:
        return {"error": str(e), "interpretation": "Impossible d'effectuer le test ADF"}


def train_forecast_model(
    series: pd.Series,
    p: int,
    d: int,
    q: int,
    P: int = 0,
    D: int = 0,
    Q: int = 0,
    m: int = 12,
    forecast_horizon: int = 12,
) -> Dict[str, Any]:
    try:
        min_obs = max(p + d + P + D + m, 10)
        if len(series) < min_obs:
            return {"success": False, "error": f"Série trop courte (>= {min_obs} obs) pour entraîner le modèle."}

        seasonal = (P > 0 or D > 0 or Q > 0)
        seasonal_order = (P, D, Q, m) if seasonal else (0, 0, 0, 0)

        model = SARIMAX(series, order=(p, d, q), seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)

        forecast_result = fitted_model.get_forecast(steps=forecast_horizon)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()

        last_date = series.index[-1]
        freq = pd.infer_freq(series.index) or "D"
        forecast_index = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=freq)[1:]

        return {
            "success": True,
            "forecast_values": forecast_mean.tolist(),
            "forecast_dates": forecast_index.astype(str).tolist(),
            "lower_ci": forecast_ci.iloc[:, 0].tolist(),
            "upper_ci": forecast_ci.iloc[:, 1].tolist(),
            "in_sample_pred": fitted_model.fittedvalues.tolist(),
            "in_sample_dates": series.index.astype(str).tolist(),
            "observed_values": series.tolist(),
            "residuals": fitted_model.resid.tolist(),
            "model_info": {
                "aic": float(fitted_model.aic),
                "bic": float(fitted_model.bic),
                "model_type": "SARIMA" if seasonal else "ARIMA",
                "order": f"({p},{d},{q})",
                "seasonal_order": f"({P},{D},{Q},{m})" if seasonal else "None",
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def generate_ai_report(
    analysis_data: Dict[str, Any],
    report_mode: str = "court",
    model_type: str = "gpt-5-mini",
) -> str:
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "⚠️ Clé API OpenAI manquante. Configure OPENAI_API_KEY sur Render."

        if report_mode == "long":
            prompt = f"""Vous êtes un expert en analyse de séries temporelles. Rapport détaillé.

DONNÉES:
{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Inclure:
1) Description dataset
2) Interprétation (tendance/saisonnalité/résidus)
3) Analyse prévisions + incertitude
4) Recos techniques (validation, résidus, amélioration)
"""
        else:
            prompt = f"""Vous êtes un expert en séries temporelles. Rapport concis et actionnable.

DONNÉES:
{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Inclure:
1) Résumé dataset (3-4 phrases)
2) Insights (tendance, saisonnalité)
3) Recos (5-6 phrases)
"""

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=model_type,
            messages=[
                {"role": "system", "content": "Expert en analyse de séries temporelles et modélisation statistique."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Erreur rapport IA: {str(e)}"


# =====================================================
# ROUTES
# =====================================================
@api_router.get("/")
async def api_root():
    return {"message": "Hello World"}


@api_router.post("/status")
async def create_status_check(status_check: StatusCheckCreate):
    new_check = StatusCheck(client_name=status_check.client_name)
    check_dict = new_check.model_dump()
    check_dict["timestamp"] = check_dict["timestamp"].isoformat()
    await db.status_checks.insert_one(check_dict)
    return new_check


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for check in status_checks:
        if isinstance(check.get("timestamp"), str):
            check["timestamp"] = datetime.fromisoformat(check["timestamp"])
    return status_checks


@api_router.post("/timeseries/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".csv", ".xlsx", ".xls")):
            raise HTTPException(status_code=400, detail="Format non supporté. Utilisez CSV/Excel (.csv, .xlsx, .xls)")

        content = await file.read()
        df = parse_dataframe(content, file.filename)

        file_id = str(uuid.uuid4())

        doc = {
            "file_id": file_id,
            "filename": file.filename,
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "columns": df.columns.tolist(),
            "n_rows": int(len(df)),
            "data": df.to_json(orient="split", date_format="iso"),
        }

        await db.timeseries_files.insert_one(doc)

        return {
            "file_id": file_id,
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "n_rows": int(len(df)),
            "preview": df.head(5).to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
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
            "frequency": pd.infer_freq(series.index) or "Non détectée",
        }

        return {
            "original": {"dates": series.index.astype(str).tolist(), "values": series.tolist()},
            "stl": stl_result,
            "adf": adf_result,
            "summary": summary,
        }
    except HTTPException:
        raise
    except Exception as e:
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
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/timeseries/ai-report")
async def generate_report(request: AIReportRequest):
    try:
        report = await generate_ai_report(request.analysis_data, request.report_mode, request.model_type)
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Include router
app.include_router(api_router)


# Root endpoint
@app.get("/")
async def main_root():
    return {"message": "TSA API - Time Series Analysis", "docs": "/docs"}
