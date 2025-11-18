# main.py -- WERSJA: OSTATECZNA FINALNA (Poprawione Nazwy + Pełen Odczyt)

import h5py
import numpy as np
import os
import logging
from typing import Optional
from datetime import datetime, timedelta, timezone 

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
CONTENT_DIR = os.path.join(BASE_DIR, "content")


# HDF5_FILE = os.path.join(BASE_DIR, "..", "KALMAN_FILTER_SDE_LAYERLESS_MODEL", "pure_sde_data", "VISUALIZATION_DATA.hdf5")

HDF5_FILE = os.path.join(BASE_DIR, "..", "BACKEND", "KALMAN_FILTER_SDE_LAYERLESS_MODEL", "pure_sde_data", "VISUALIZATION_DATA.hdf5")


INITIAL_POINTS = 2880
HISTORY_LIMIT_DAYS = 3

app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATES_DIR)

def clean_nan_values(data_list):
    return [None if pd.isna(val) else val for val in data_list]

@app.get("/static/{filename}")
async def get_static_file(filename: str):
    file_path = os.path.join(STATIC_DIR, filename)
    if os.path.exists(file_path):
        media_type = "text/css" if filename.endswith(".css") else "application/javascript"
        return FileResponse(file_path, media_type=media_type)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    try:
        with open(os.path.join(CONTENT_DIR, "description.txt"), "r", encoding="utf-8") as f:
            description = f.read()
    except FileNotFoundError:
        description = "Description not available."
    return templates.TemplateResponse("index.html", {"request": request, "description": description})

@app.get("/history", response_class=JSONResponse)
async def get_history(start_ts: Optional[float] = None, end_ts: Optional[float] = None):
    try:
        with h5py.File(HDF5_FILE, 'r') as f:
            if 'Time' not in f:
                return JSONResponse(status_code=404, content={"message": "Data not found."})

            all_timestamps = f['Time'][:]
            total_rows = len(all_timestamps)
           
            start_idx, end_idx = 0, total_rows

            limit_dt = datetime.now(timezone.utc) - timedelta(days=HISTORY_LIMIT_DAYS)
            limit_ts_int = np.int64(limit_dt.timestamp() * 1000)
            history_limit_idx = np.searchsorted(all_timestamps, limit_ts_int, side='left')

            if start_ts is not None and end_ts is not None:
                start_ts_int = np.int64(start_ts)
                req_start_idx = np.searchsorted(all_timestamps, start_ts_int, side='left')
                start_idx = max(req_start_idx, history_limit_idx)
                end_ts_int = np.int64(end_ts)
                end_idx = np.searchsorted(all_timestamps, end_ts_int, side='right')
            else:
                initial_points_idx = max(0, total_rows - INITIAL_POINTS)
                start_idx = max(initial_points_idx, history_limit_idx)

            if start_idx >= end_idx:
                return JSONResponse(content={"Time": []})
           
            data_slice = slice(start_idx, end_idx)
            timestamps_python_list = [int(ts) for ts in all_timestamps[data_slice]]

            # --- OSTATECZNA, POPRAWIONA SEKCJA WCZYTYWANIA DANYCH ---
            response_data = {"Time": timestamps_python_list}
            
            # Lista wszystkich kolumn, które chcemy wczytać
            columns_to_load = [
                "Actual_BTC_Price", "KAMA_Actual_BTC_Price", "Forecast",
                "ARIMAX_Forecast", "KAMA_Forecast", "Accuracy_ARIMAX", "Accuracy_Final"
            ]
            
            for col in columns_to_load:
                if col in f:
                    response_data[col] = clean_nan_values(f[col][data_slice].tolist())
            
            logger.info(f"Served {len(timestamps_python_list)} records.")
            return JSONResponse(content=response_data)
           
    except FileNotFoundError:
        logger.warning(f"File not found: {HDF5_FILE}")
        return JSONResponse(status_code=404, content={"message": "Data file not found."})
    except Exception as e:
        logger.exception(f"Error in /history: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal server error."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8050, reload=True)
