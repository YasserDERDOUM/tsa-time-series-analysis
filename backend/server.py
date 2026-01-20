# server.py
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient

import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Tuple
import uuid
from datetime import datetime, timezone
import io
import json
import warnings

import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

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
# JSON SAFE (fix numpy/pandas types)
# =====================================================
def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serializable Python types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]

    # pandas
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]

    # fallback
    try:
        if hasattr(obj, "item"):
            return to_jsonable(obj.item())
    except Exception:
        pass

    return str(obj)


# =====================================================
# MONGODB (safe URI)
# =====================================================
def sanitize_mongo_uri(uri: str) -> str:
    """
    Ensure username/password in MongoDB URI are RFC3986-escaped.
    Fixes passwords with special chars like @ : / ? # %
    """
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
# CORS
# =====================================================
# Example (Render env):
# CORS_ORIGINS=https://tsafr0.netlify.app,http://localhost:3000
cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
origins = [o.strip().rstrip("/") for o in cors_origins_raw.split(",") if o.strip()]

logger.info("CORS origins=%s | allow_credentials=False | allow_origin_regex=netlify", origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https://.*\.netlify\.app$",  # allow all netlify deploy URLs
    allow_credentials=False,  # IMPORTANT: must be False when using regex/wildcards
    allow_methods=["*"],
    allow_headers=["*"],
)

# preflight (helps some environments)
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
    analysis_data: Dict[str, Any]  # can be compact payload from frontend OR full analyze payload
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


def prepare_timeseries(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    duplicate_strategy: str = "mean",
) -> pd.Series:
    """
    - parse date column
    - coerce value to numeric
    - drop NaNs
    - sort by date
    - handle duplicates (mean/sum)
    - set index
    """
    try:
        if date_col not in df.columns or value_col not in df.columns:
            raise HTTPException(status_code=400, detail="Colonnes date/valeur introuvables dans le fichier.")

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)

        if duplicate_strategy == "mean":
            df = df.groupby(date_col)[value_col].mean().reset_index()
        elif duplicate_strategy == "sum":
            df = df.groupby(date_col)[value_col].sum().reset_index()
        else:
            # fallback
            df = df.groupby(date_col)[value_col].mean().reset_index()

        df.set_index(date_col, inplace=True)
        s = df[value_col].astype(float)

        # remove tz if any
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_convert(None)

        return s
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de préparation des données: {str(e)}")


def infer_period_from_freq(freq: Optional[str]) -> int:
    """
    Map inferred frequency to a reasonable seasonal period.
    Note: infer_freq returns things like 'D', 'W-SUN', 'MS', 'M', 'QS-OCT', etc.
    """
    if not freq:
        return 12
    f = str(freq).upper()

    # Monthly
    if "M" in f:
        return 12
    # Quarterly
    if "Q" in f:
        return 4
    # Weekly
    if "W" in f:
        return 52
    # Daily
    if f.startswith("D"):
        return 7

    return 12


def safe_infer_freq(index: pd.DatetimeIndex) -> Optional[str]:
    try:
        return pd.infer_freq(index)
    except Exception:
        return None


def perform_stl(series: pd.Series, period: Optional[int] = None) -> Dict[str, Any]:
    """
    STL needs:
    - enough data (>= 2*period is a good rule of thumb)
    - regular-ish frequency
    """
    try:
        freq = safe_infer_freq(series.index)
        p = period or infer_period_from_freq(freq)

        if len(series) < 2 * p:
            return {
                "success": False,
                "error": f"STL impossible: série trop courte (besoin >= {2*p} obs, obtenu {len(series)}).",
                "period": int(p),
                "freq": freq,
            }

        # 'seasonal' in statsmodels STL is window length (odd >= 3), NOT the period.
        # We keep a standard window ~ period (odd).
        seasonal_window = int(p if p % 2 == 1 else p + 1)
        seasonal_window = max(seasonal_window, 7)  # decent minimum

        stl = STL(series, period=p, seasonal=seasonal_window, robust=True)
        res = stl.fit()

        # meta
        seasonal = res.seasonal
        resid = res.resid
        s_var = float(np.var(series.values))
        sea_var = float(np.var(seasonal.values))
        res_var = float(np.var(resid.values))
        seasonal_strength = sea_var / s_var if s_var > 0 else None
        resid_ratio = res_var / s_var if s_var > 0 else None
        sea_amp = float(np.nanmax(seasonal.values) - np.nanmin(seasonal.values))

        return {
            "success": True,
            "period": int(p),
            "freq": freq,
            # arrays for plotting
            "dates": series.index.astype(str).tolist(),
            "trend": res.trend.tolist(),
            "seasonal": seasonal.tolist(),
            "resid": resid.tolist(),
            # meta for IA
            "meta": {
                "seasonal_amplitude": sea_amp,
                "seasonal_strength": seasonal_strength,
                "resid_ratio": resid_ratio,
                "seasonal_window": seasonal_window,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "period": int(period or 12),
            "freq": safe_infer_freq(series.index),
        }


def adf_summary(series: pd.Series) -> Dict[str, Any]:
    try:
        x = series.dropna()
        if len(x) < 10:
            return {"error": "ADF: série trop courte (<10)."}
        adf_stat, p_value, usedlag, nobs, critical_values, _ = adfuller(x)
        is_stationary = bool(p_value < 0.05)
        return {
            "adf_statistic": float(adf_stat),
            "p_value": float(p_value),
            "used_lag": int(usedlag),
            "n_obs": int(nobs),
            "critical_values": {k: float(v) for k, v in critical_values.items()},
            "is_stationary": is_stationary,
        }
    except Exception as e:
        return {"error": str(e)}


def compute_stationarity_profile(series: pd.Series, m: int) -> Dict[str, Any]:
    """
    ADF on:
      - original
      - diff(1)
      - seasonal diff(m)
      - diff(1) + seasonal diff(m)
    And propose recommended d and D.
    """
    orig = adf_summary(series)
    d1 = adf_summary(series.diff(1))
    sd = adf_summary(series.diff(m)) if m and m > 1 else {"error": "m invalide"}
    d1sd = adf_summary(series.diff(1).diff(m)) if m and m > 1 else {"error": "m invalide"}

    # propose d
    recommended_d = 0
    if isinstance(orig.get("is_stationary"), bool) and not orig["is_stationary"]:
        recommended_d = 1
        if isinstance(d1.get("is_stationary"), bool) and not d1["is_stationary"]:
            # don't overdo, but suggest 2 if still not stationary
            recommended_d = 2

    # propose D
    recommended_D = 0
    # if seasonality suspected (m>1) and seasonal diff helps
    if m and m > 1:
        if isinstance(orig.get("is_stationary"), bool) and not orig["is_stationary"]:
            # if seasonal diff becomes stationary while d1 isn't, recommend D=1
            if isinstance(sd.get("is_stationary"), bool) and sd["is_stationary"]:
                recommended_D = 1
            elif isinstance(d1sd.get("is_stationary"), bool) and d1sd["is_stationary"]:
                recommended_D = 1

    return {
        "m": int(m),
        "adf": {
            "original": orig,
            "diff1": d1,
            "seasonal_diff": sd,
            "diff1_plus_seasonal": d1sd,
        },
        "recommended": {"d": int(recommended_d), "D": int(recommended_D)},
    }


def compute_time_profile(series: pd.Series) -> Dict[str, Any]:
    s = series.dropna().astype(float)
    n = len(s)
    if n < 3:
        return {"error": "Série trop courte pour analyser les variations."}

    diff = s.diff().dropna()
    pct = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    start_val = float(s.iloc[0])
    end_val = float(s.iloc[-1])
    abs_change = end_val - start_val
    pct_change = (abs_change / start_val * 100.0) if start_val != 0 else None

    top_up = diff.nlargest(3)
    top_down = diff.nsmallest(3)

    def _movements(x: pd.Series):
        return [{"date": str(idx), "delta": float(val)} for idx, val in x.items()]

    step = max(1, n // 20)
    sample = [{"date": str(s.index[i]), "value": float(s.iloc[i])} for i in range(0, n, step)][:20]

    return {
        "n_points": int(n),
        "start": {"date": str(s.index[0]), "value": start_val},
        "end": {"date": str(s.index[-1]), "value": end_val},
        "global_change": {"absolute": float(abs_change), "percent": float(pct_change) if pct_change is not None else None},
        "diff_stats": {
            "mean_delta": float(diff.mean()) if len(diff) else None,
            "std_delta": float(diff.std()) if len(diff) else None,
        },
        "pct_stats": {
            "mean_pct": float(pct.mean() * 100.0) if len(pct) else None,
            "std_pct": float(pct.std() * 100.0) if len(pct) else None,
        },
        "volatility": {"std": float(s.std()), "cv": float(s.std() / s.mean()) if s.mean() != 0 else None},
        "top_increases": _movements(top_up),
        "top_decreases": _movements(top_down),
        "sample_points": sample,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # MAPE (safe)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(err) / denom) * 100.0)

    return {"mae": mae, "rmse": rmse, "mape": mape}


def ljung_box_pvalue(residuals: pd.Series, lags: Optional[int] = None) -> Optional[float]:
    try:
        r = pd.Series(residuals).dropna()
        if len(r) < 20:
            return None
        L = lags or min(24, max(10, len(r) // 5))
        out = acorr_ljungbox(r, lags=[L], return_df=True)
        return float(out["lb_pvalue"].iloc[0])
    except Exception:
        return None


def fit_sarimax(
    series: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> Any:
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def backtest_one_step(
    series: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    test_size: int,
) -> Dict[str, Any]:
    """
    Simple backtest:
    - train = series[:-test_size]
    - forecast steps = test_size
    - compute metrics against last test_size points
    """
    try:
        if test_size < 3:
            return {"success": False, "error": "test_size trop petit"}

        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]

        fitted = fit_sarimax(train, order, seasonal_order)
        fc = fitted.get_forecast(steps=test_size).predicted_mean

        metrics = compute_metrics(test.values, fc.values)
        return {
            "success": True,
            "test_size": int(test_size),
            "metrics": metrics,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def generate_ai_report(analysis_data: Dict[str, Any], report_mode: str, model_type: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ OPENAI_API_KEY manquante sur le backend (Render). Ajoute-la dans Environment Variables."

    # Keep it structured: forces the model to talk about STL, differencing, model choice, forecast quality, residuals.
    prompt = f"""
Tu es un expert en séries temporelles (STL / ADF / ARIMA / SARIMA / diagnostics).
Objectif: analyser la série, expliquer les prévisions, recommander le meilleur modèle et donner des améliorations concrètes.

DONNÉES (résumé & diagnostics) :
{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

RÈGLES :
- Explique toujours le POURQUOI (pas seulement des chiffres).
- Si STL échoue, explique précisément la cause probable + comment corriger (fréquence irrégulière, série trop courte, period incorrect, etc.).
- Compare ARIMA vs SARIMA et JUSTIFIE (saisonnalité + stationnarité + backtest + résidus).
- Donne des recommandations concrètes (grid-search, backtesting rolling, log/Box-Cox, exogènes SARIMAX, resampling, outliers).

PLAN OBLIGATOIRE :
1) Lecture dataset: période, fréquence, qualité (doublons, manquants, stratégie).
2) Analyse série: tendance (hausse/baisse), variations, volatilité, ruptures.
3) STL: interprétation (tendance/saisonnalité/résidu) + intensité de saisonnalité (strength, amplitude, resid_ratio).
4) Stationnarité & différenciation: ADF original/diff/seasonal diff. Recommande d et D (et explique).
5) Modélisation: quel modèle est le plus adapté (ARIMA vs SARIMA) et pourquoi.
6) Prévisions: trajectoire, incertitude (IC), risques, cohérence métier.
7) Diagnostics résidus: Ljung-Box (si fourni), biais, conclusion sur la fiabilité.
8) Améliorations (max 6 bullets): ultra concrètes et actionnables.

Style: {'concis (10-15 lignes max)' if report_mode=='court' else 'détaillé, structuré avec titres'}.
"""

    client = AsyncOpenAI(api_key=api_key)

    # OpenAI python SDK v1: use max_completion_tokens
    max_completion = 900 if report_mode == "court" else 1800

    resp = await client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": "Tu es un expert en analyse de séries temporelles et modélisation statistique."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_completion_tokens=max_completion,
    )
    return resp.choices[0].message.content


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
        if isinstance(check.get("timestamp"), str):
            check["timestamp"] = datetime.fromisoformat(check["timestamp"])
    return status_checks


@api_router.post("/timeseries/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".csv", ".xlsx", ".xls")):
            raise HTTPException(status_code=400, detail="Format non supporté. Utilisez CSV ou Excel (.xlsx, .xls).")

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

        return to_jsonable(
            {
                "file_id": file_id,
                "filename": file.filename,
                "columns": df.columns.tolist(),
                "n_rows": int(len(df)),
                "preview": df.head(5).to_dict(orient="records"),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("upload_file error")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@api_router.post("/timeseries/analyze")
async def analyze_timeseries(request: ColumnSelectionRequest):
    """
    Returns:
      - original series (dates, values) for plotting
      - STL (arrays + meta)
      - ADF profile (original/diff/seasonal)
      - summary
      - time_profile (variation vs time)
    """
    try:
        doc = await db.timeseries_files.find_one({"file_id": request.file_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Fichier non trouvé")

        df = pd.read_json(io.StringIO(doc["data"]), orient="split")
        series = prepare_timeseries(df, request.date_column, request.value_column, request.duplicate_strategy)

        freq = safe_infer_freq(series.index)
        m = infer_period_from_freq(freq)  # default seasonal period for diagnostics

        stl = perform_stl(series)
        stationarity_profile = compute_stationarity_profile(series, m=m)
        time_profile = compute_time_profile(series)

        summary = {
            "n_observations": int(len(series)),
            "start_date": str(series.index[0]),
            "end_date": str(series.index[-1]),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "missing_values": int(series.isna().sum()),
            "frequency": str(freq or "Non détectée"),
            "suggested_seasonal_period_m": int(m),
            "duplicate_strategy": request.duplicate_strategy,
        }

        payload = {
            "original": {"dates": series.index.astype(str).tolist(), "values": series.tolist()},
            "stl": stl,  # arrays + meta
            "stationarity": stationarity_profile,
            "summary": summary,
            "time_profile": time_profile,
        }

        return to_jsonable(payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("analyze_timeseries error")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@api_router.post("/timeseries/forecast")
async def forecast_timeseries(request: ForecastRequest):
    """
    Train chosen ARIMA/SARIMA and return:
      - forecasts + CI
      - in-sample fitted values
      - residuals
      - diagnostics: Ljung-Box p-value
      - backtest metrics
    """
    try:
        doc = await db.timeseries_files.find_one({"file_id": request.file_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Fichier non trouvé")

        df = pd.read_json(io.StringIO(doc["data"]), orient="split")
        series = prepare_timeseries(df, request.date_column, request.value_column, request.duplicate_strategy)

        seasonal = bool(request.seasonal) and (request.P > 0 or request.D > 0 or request.Q > 0)
        seasonal_order = (int(request.P), int(request.D), int(request.Q), int(request.m)) if seasonal else (0, 0, 0, 0)
        order = (int(request.p), int(request.d), int(request.q))

        min_obs = max(sum(order) + (seasonal_order[0] + seasonal_order[1] + seasonal_order[2]) + seasonal_order[3], 12)
        if len(series) < min_obs:
            return to_jsonable({"success": False, "error": f"Série trop courte (>= {min_obs} obs) pour ce modèle."})

        # Fit on full series
        fitted = fit_sarimax(series, order=order, seasonal_order=seasonal_order)

        # Forecast
        fc = fitted.get_forecast(steps=int(request.forecast_horizon))
        mean = fc.predicted_mean
        ci = fc.conf_int()

        # Build future index
        freq = safe_infer_freq(series.index) or "D"
        forecast_index = pd.date_range(start=series.index[-1], periods=int(request.forecast_horizon) + 1, freq=freq)[1:]

        residuals = fitted.resid
        lb_p = ljung_box_pvalue(residuals)

        # Backtest (simple holdout)
        test_size = min(max(12, int(request.forecast_horizon)), max(3, len(series) // 5))
        backtest = backtest_one_step(series, order=order, seasonal_order=seasonal_order, test_size=int(test_size))

        result = {
            "success": True,
            "forecast_values": mean.tolist(),
            "forecast_dates": forecast_index.astype(str).tolist(),
            "lower_ci": ci.iloc[:, 0].tolist(),
            "upper_ci": ci.iloc[:, 1].tolist(),
            "in_sample_pred": fitted.fittedvalues.tolist(),
            "in_sample_dates": series.index.astype(str).tolist(),
            "observed_values": series.tolist(),
            "residuals": residuals.tolist(),
            "diagnostics": {
                "ljung_box_pvalue": lb_p,  # if <0.05 => residuals autocorrelated (bad)
            },
            "backtest": backtest,
            "model_info": {
                "aic": float(fitted.aic),
                "bic": float(fitted.bic),
                "model_type": "SARIMA" if seasonal else "ARIMA",
                "order": f"({order[0]},{order[1]},{order[2]})",
                "seasonal_order": f"({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]},{seasonal_order[3]})"
                if seasonal
                else "None",
                "freq": str(freq),
            },
        }
        return to_jsonable(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("forecast_timeseries error")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/timeseries/ai-report")
async def generate_report(request: AIReportRequest):
    """
    Front can send:
      - analysis_data from analyze endpoint + forecast endpoint merged
    We will shrink it for the LLM so it remains cheap and still "understands" the series.
    """
    try:
        data = request.analysis_data or {}

        # Extract key blocks if present
        summary = data.get("summary")
        stl = data.get("stl")
        stationarity = data.get("stationarity") or data.get("adf")  # keep compatibility
        time_profile = data.get("time_profile")

        forecast = data.get("forecast") or data.get("forecast_results") or {}
        model_info = forecast.get("model_info") or data.get("model_info")
        backtest = forecast.get("backtest") or data.get("backtest")
        diagnostics = forecast.get("diagnostics") or data.get("diagnostics")

        # If frontend sent "forecast_summary" only (older), keep it
        forecast_summary = forecast.get("forecast_summary") or data.get("forecast_summary")

        # STL meta only (avoid huge arrays)
        stl_meta = None
        if isinstance(stl, dict):
            stl_meta = {
                "success": stl.get("success"),
                "period": stl.get("period"),
                "freq": stl.get("freq"),
                "error": stl.get("error"),
                "meta": (stl.get("meta") or {}),
            }

        # Stationarity compact
        stationarity_compact = stationarity
        if isinstance(stationarity, dict) and "adf" in stationarity:
            stationarity_compact = {
                "m": stationarity.get("m"),
                "recommended": stationarity.get("recommended"),
                "adf": stationarity.get("adf"),
            }

        llm_payload = {
            "summary": summary,
            "time_profile": time_profile,
            "stl": stl_meta,
            "stationarity": stationarity_compact,
            "forecast": {
                "model_info": model_info,
                "backtest": backtest,
                "diagnostics": diagnostics,
                "forecast_summary": forecast_summary,
            },
        }

        llm_payload = to_jsonable(llm_payload)

        report = await generate_ai_report(llm_payload, request.report_mode, request.model_type)
        return {"report": report}

    except Exception as e:
        logger.exception("ai-report error")
        raise HTTPException(status_code=500, detail=str(e))


# include router
app.include_router(api_router)

# root endpoint
@app.get("/")
async def main_root():
    return {"message": "TSA API - Time Series Analysis", "docs": "/docs"}
