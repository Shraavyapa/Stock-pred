import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from model import train_lstm_model, predict_future_prices, evaluate_model, detect_currency
import ssl
import time

# SSL configuration
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

app = FastAPI(
    title="AI Stock Predictor API",
    description="API for stock price prediction and lookup using LSTM models and yFinance data",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    ticker: str
    start_date: str  # Format: YYYY-MM-DD
    end_date: str    # Format: YYYY-MM-DD
    prediction_days: int
    time_steps: int

class NewsItem(BaseModel):
    title: str
    source: str
    date: Optional[str]
    link: Optional[str]

class StockInfoResponse(BaseModel):
    ticker: str
    company_name: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    country: Optional[str]
    website: Optional[str]
    description: Optional[str]
    current_price: Optional[float]
    previous_close: Optional[float]
    market_cap: Optional[float]
    week_high_52: Optional[float]
    week_low_52: Optional[float]
    volume: Optional[int]
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    currency_code: str
    currency_symbol: str
    historical_data: Optional[List[Dict[str, Any]]]
    news: List[NewsItem]
    institutional_holders: Optional[List[Dict[str, Any]]]
    recommendations: Optional[List[Dict[str, Any]]]

class PredictionResponse(BaseModel):
    ticker: str
    currency_code: str
    currency_symbol: str
    historical_data: List[Dict[str, Any]]
    future_predictions: List[Dict[str, Any]]
    metrics: Dict[str, float]
    current_price: float
    predicted_price: float
    price_change_pct: float
    recommendation: str

def fetch_stock_data(ticker: str, start: str, end: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start, end=end, interval='1d', auto_adjust=False)
            if not data.empty:
                data.index = pd.to_datetime(data.index)
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_cols):
                    raise ValueError("Missing required columns in data")
                for col in ['Open', 'High', 'Low']:
                    if data[col].isnull().all():
                        data[col] = data['Close']
                data['20_MA'] = data['Close'].rolling(window=20).mean().fillna(method='bfill')
                stock_info = stock.info
                return data, stock_info
            raise ValueError("Empty data returned")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise HTTPException(status_code=500, detail=f"Failed to fetch data for {ticker}: {str(e)}")

@app.get("/stock/{ticker}", response_model=StockInfoResponse)
async def get_stock_info(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        currency_code, currency_symbol = detect_currency(ticker, info)

        if not info or (info.get('symbol') and info.get('symbol') != ticker and not ticker.endswith(('.NS', '.BO'))):
            raise HTTPException(status_code=404, detail=f"Invalid ticker: {ticker}")

        # Historical data (1 year)
        hist_data = stock.history(period="1y")
        historical_data = (
            hist_data.reset_index().to_dict('records') if not hist_data.empty else []
        )

        # News
        news_items = []
        try:
            news = None
            for attempt in range(3):
                try:
                    news = stock.news
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    raise e
            print(f"News data for {ticker}: {news}")  # Backend debug
            if news and isinstance(news, list):
                for i, article in enumerate(news[:5], 1):
                    try:
                        content = article.get('content') or {}
                        title = content.get('title', 'N/A')
                        publisher = content.get('provider', {}).get('displayName', '')
                        pub_date = content.get('pubDate')
                        click_url = content.get('clickThroughUrl')
                        canon_url = content.get('canonicalUrl')
                        link = None
                        if click_url and isinstance(click_url, dict):
                            link = click_url.get('url')
                        elif canon_url and isinstance(canon_url, dict):
                            link = canon_url.get('url')
                        if not title or title == 'N/A' or not publisher or not link:
                            continue
                        date_str = None
                        if pub_date:
                            try:
                                date_str = datetime.strptime(pub_date, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
                            except ValueError:
                                date_str = str(pub_date)
                        news_items.append(NewsItem(
                            title=title,
                            source=publisher,
                            date=date_str,
                            link=link
                        ))
                    except Exception as e:
                        print(f"Skipping article {i} for {ticker}: {str(e)}")
                        continue
        except Exception as e:
            print(f"Error fetching news for {ticker}: {str(e)}")

        # Institutional Holders
        institutional_holders = []
        try:
            institutions = stock.institutional_holders
            if institutions is not None and not institutions.empty:
                institutional_holders = institutions.to_dict('records')[:10]
        except Exception:
            pass

        # Recommendations
        recommendations = []
        try:
            recs = stock.recommendations
            if recs is not None and not recs.empty:
                recommendations = recs.tail(5).to_dict('records')
        except Exception:
            pass

        return StockInfoResponse(
            ticker=ticker,
            company_name=info.get('longName'),
            sector=info.get('sector'),
            industry=info.get('industry'),
            country=info.get('country'),
            website=info.get('website'),
            description=info.get('longBusinessSummary'),
            current_price=info.get('currentPrice') or info.get('regularMarketPrice'),
            previous_close=info.get('previousClose') or info.get('regularMarketPreviousClose'),
            market_cap=info.get('marketCap'),
            week_high_52=info.get('fiftyTwoWeekHigh'),
            week_low_52=info.get('fiftyTwoWeekLow'),
            volume=info.get('volume') or info.get('regularMarketVolume'),
            pe_ratio=info.get('trailingPE'),
            dividend_yield=info.get('dividendYield'),
            beta=info.get('beta'),
            currency_code=currency_code,
            currency_symbol=currency_symbol,
            historical_data=historical_data,
            news=news_items,
            institutional_holders=institutional_holders,
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data for {ticker}: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock_price(request: PredictionRequest):
    try:
        start_date = datetime.strptime(request.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(request.end_date, '%Y-%m-%d').date()
        ticker = request.ticker.upper().strip()

        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        if request.prediction_days not in [7, 14, 30, 60]:
            raise HTTPException(status_code=400, detail="Invalid prediction days")
        if request.time_steps not in [30, 60, 90]:
            raise HTTPException(status_code=400, detail="Invalid time steps")

        stock_data, stock_info = fetch_stock_data(ticker, start_date, end_date)
        currency_code, currency_symbol = detect_currency(ticker, stock_info)

        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        required_days = request.time_steps + 50
        if len(stock_data) < required_days:
            raise HTTPException(status_code=400, detail=f"Insufficient data: need {required_days} days, got {len(stock_data)}")

        model, scaler, X_test, y_test, df_clean, feature_columns, history = train_lstm_model(
            stock_data, time_steps=request.time_steps
        )

        future_predictions, future_dates = predict_future_prices(
            model, scaler, stock_data, feature_columns, days=request.prediction_days, time_steps=request.time_steps
        )
        future_predictions = np.nan_to_num(future_predictions, nan=0.0, posinf=0.0, neginf=0.0).flatten()

        metrics = evaluate_model(model, scaler, X_test, y_test, feature_columns)
        current_price = float(stock_data['Close'].iloc[-1])
        predicted_price = float(future_predictions[-1]) if len(future_predictions) > 0 else 0.0
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100 if current_price != 0 else 0

        recommendation = (
            "Strong Buy" if price_change_pct > 5 else
            "Buy" if price_change_pct > 2 else
            "Hold" if price_change_pct > -2 else
            "Sell" if price_change_pct > -5 else
            "Strong Sell"
        )

        historical_data = stock_data.reset_index().to_dict('records')
        future_predictions_data = [
            {"Date": future_dates[i].strftime('%Y-%m-%d'), "Predicted_Close": float(future_predictions[i])}
            for i in range(len(future_predictions))
        ]

        return PredictionResponse(
            ticker=ticker,
            currency_code=currency_code,
            currency_symbol=currency_symbol,
            historical_data=historical_data,
            future_predictions=future_predictions_data,
            metrics={
                "mae": float(metrics.get("mae", 0.0)),
                "rmse": float(metrics.get("rmse", 0.0)),
                "mape": float(metrics.get("mape", 0.0)),
                "r2": float(metrics.get("r2", 0.0))
            },
            current_price=current_price,
            predicted_price=predicted_price,
            price_change_pct=price_change_pct,
            recommendation=recommendation
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error for {ticker}: {str(e)}")