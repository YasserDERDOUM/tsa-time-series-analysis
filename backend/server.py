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
    netloc = parts.netloc

    if "@" not in netloc:
        return uri

    userinfo, hostpart = netloc.rsplit("@", 1)

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
# FASTAPI
# =====================================================
app = FastAPI()
api_router = APIRouter(prefix="/api")


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
    """Parse CSV or Excel file and return DataFrame"""
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
    df: pd.DataFrame, date_col: str, value_col: str, duplicate_strategy: str = "mean"
) -> pd.Series:
    """Prepare time series data: parse dates, handle duplicates, remove NaN"""
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, value_col])
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col])
        df = df.sort_values(date_col)

        if duplicate_strategy == "mean":
            df = df.groupby(date_col)[value_col].mean().reset_index()
        elif duplicate_strategy == "sum":
            df = df.groupby(date_col)[value_col].sum().reset_index()

        df.set_index(date_col, inplace=True)
        return df[value_col]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de préparation des données: {str(e)}")


def perform_stl_decomposition(series: pd.Series, period: int = None) -> Dict[str, Any]:
    """Perform STL decomposition"""
    try:
        if period is None:
            if len(series) > 2:
                inferred_freq = pd.infer_freq(series.index)
                if inferred_freq:
                    period = {"D": 7, "M": 12, "Q": 4, "Y": 1, "W": 52}.get(inferred_freq[0], 12)
                else:
                    period = 12
            else:
                period = 12

        if len(series) < 2 * period:
            return {
                "success": False,
                "error": f"Série trop courte pour décomposition STL (besoin d'au moins {2 * period} observations, seulement {len(series)} disponibles)",
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
        return {"success": False, "error": str(e), "period": period if period else 12}


def perform_adf_test(series: pd.Series) -> Dict[str, Any]:
    """Perform Augmented Dickey-Fuller test for stationarity"""
    try:
        result = adfuller(series.dropna())
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
            "is_stationary": is_stationary,
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
    """Train ARIMA/SARIMA model and generate forecasts"""
    try:
        min_obs = max(p + d + P + D + m, 10)
        if len(series) < min_obs:
            return {"success": False, "error": f"Série trop courte pour entraîner le modèle (besoin d'au moins {min_obs} observations)"}

        seasonal = P > 0 or D > 0 or Q > 0
        seasonal_order = (P, D, Q, m) if seasonal else (0, 0, 0, 0)

        model = SARIMAX(series, order=(p, d, q), seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)

        forecast_result = fitted_model.get_forecast(steps=forecast_horizon)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()

        in_sample_pred = fitted_model.fittedvalues
        residuals = fitted_model.resid

        model_info = {
            "aic": float(fitted_model.aic),
            "bic": float(fitted_model.bic),
            "model_type": "SARIMA" if seasonal else "ARIMA",
            "order": f"({p},{d},{q})",
            "seasonal_order": f"({P},{D},{Q},{m})" if seasonal else "None",
        }

        last_date = series.index[-1]
        freq = pd.infer_freq(series.index) or "D"
        forecast_index = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=freq)[1:]

        return {
            "success": True,
            "forecast_values": forecast_mean.tolist(),
            "forecast_dates": forecast_index.astype(str).tolist(),
            "lower_ci": forecast_ci.iloc[:, 0].tolist(),
            "upper_ci": forecast_ci.iloc[:, 1].tolist(),
            "in_sample_pred": in_sample_pred.tolist(),
            "in_sample_dates": series.index.astype(str).tolist(),
            "observed_values": series.tolist(),
            "residuals": residuals.tolist(),
            "model_info": model_info,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def generate_ai_report(
    analysis_data: Dict[str, Any],
    report_mode: str = "court",
    model_type: str = "gpt-4o-mini",
) -> str:
    """Generate AI report using OpenAI API"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "⚠️ Clé API OpenAI manquante. Veuillez configurer OPENAI_API_KEY"

        if report_mode == "long":
            prompt = f"""Vous êtes un expert en analyse de séries temporelles. Analysez les données suivantes et fournissez un rapport détaillé et professionnel.

DONNÉES D'ANALYSE:
{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Votre rapport doit inclure:
1. Description détaillée du dataset (colonnes, période, fréquence, qualité, valeurs manquantes)
2. Interprétation approfondie de chaque graphique (série originale, tendance, saisonnalité, résidus)
3. Analyse détaillée des prévisions (direction, saisonnalité, incertitude, intervalles de confiance)
4. Recommandations techniques (choix ARIMA/SARIMA, paramètres, validation, résidus, améliorations)

Fournissez un rapport structuré, professionnel et actionnable."""
        else:
            prompt = f"""Vous êtes un expert en analyse de séries temporelles. Fournissez une analyse concise et professionnelle.

DONNÉES:
{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Rapport concis incluant:
1. Résumé du dataset (3-4 phrases)
2. Principaux insights (tendance, saisonnalité)
3. Qualité des prévisions et recommandations clés (5-6 phrases maximum)

Soyez direct et actionnable."""

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=model_type,
            messages=[
                {"role": "system", "content": "Vous êtes un expert en analyse de séries temporelles et en modélisation statistique."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Erreur lors de la génération du rapport IA: {str(e)}"


# =====================================================
# ROUTES
# =====================================================
@api_router.get("/")
async def root():
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
        if isinstance(check["timestamp"], str):
            check["timestamp"] = datetime.fromisoformat(check["timestamp"])
    return status_checks


@api_router.post("/timeseries/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV or Excel file for time series analysis"""
    try:
        if not file.filename.endswith((".csv", ".xlsx", ".xls")):
            raise HTTPException(status_code=400, detail="Format non supporté. Utilisez CSV ou Excel (.xlsx, .xls)")

        content = await file.read()
        df = parse_dataframe(content, file.filename)

        file_id = str(uuid.uuid4())
        doc = {
            "file_id": file_id,
            "filename": file.filename,
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "columns": df.columns.tolist(),
            "n_rows": len(df),
            "data": df.to_json(orient="split", date_format="iso"),
        }
        await db.timeseries_files.insert_one(doc)

        return {
            "file_id": file_id,
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "n_rows": len(df),
            "preview": df.head(5).to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@api_router.post("/timeseries/analyze")
async def analyze_timeseries(request: ColumnSelectionRequest):
    """Analyze time series: original plot, STL decomposition, ADF test"""
    try:
        doc = await db.timeseries_files.find_one({"file_id": request.file_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Fichier non trouvé")

        df = pd.read_json(io.StringIO(doc["data"]), orient="split")
        series = prepare_timeseries(df, request.date_column, request.value_column, request.duplicate_strategy)

        original_data = {"dates": series.index.astype(str).tolist(), "values": series.tolist()}
        stl_result = perform_stl_decomposition(series)
        adf_result = perform_adf_test(series)

        summary = {
            "n_observations": len(series),
            "start_date": str(series.index[0]),
            "end_date": str(series.index[-1]),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "missing_values": int(series.isna().sum()),
            "frequency": pd.infer_freq(series.index) or "Non détectée",
        }

        return {"original": original_data, "stl": stl_result, "adf": adf_result, "summary": summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@api_router.post("/timeseries/forecast")
async def forecast_timeseries(request: ForecastRequest):
    """Train ARIMA/SARIMA model and generate forecasts"""
    try:
        doc = await db.timeseries_files.find_one({"file_id": request.file_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Fichier non trouvé")

        df = pd.read_json(io.StringIO(doc["data"]), orient="split")
        series = prepare_timeseries(df, request.date_column, request.value_column, request.duplicate_strategy)

        result = train_forecast_model(
            series,
            request.p,
            request.d,
            request.q,
            request.P,
            request.D,
            request.Q,
            request.m,
            request.forecast_horizon,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/timeseries/ai-report")
async def generate_report(request: AIReportRequest):
    """Generate AI-powered analysis report"""
    try:
        report = await generate_ai_report(request.analysis_data, request.report_mode, request.model_type)
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# INCLUDE ROUTER
# =====================================================
app.include_router(api_router)


# =====================================================
# CORS (FINAL FIX FOR NETLIFY + RENDER)
# - no trailing slash
# - allow_credentials must be False (no cookies)
# - allow_origin_regex for Netlify preview domains
# =====================================================
cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,https://tsafr0.netlify.app",
)

origins = [o.strip().rstrip("/") for o in cors_origins.split(",") if o.strip()]
logger.info("CORS origins=%s | allow_credentials=False", origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# ROOT
# =====================================================
@app.get("/")
async def main_root():
    return {"message": "TSA API - Time Series Analysis", "docs": "/docs"}
