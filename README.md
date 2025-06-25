# AI Stock Price Predictor

This project is an interactive Streamlit web application for stock price prediction using deep learning (LSTM) and advanced technical indicators. It uses `yfinance` for data, `pandas-ta` for technical features, and a simple LSTM model for forecasting.

## Features

- **Historical Data Fetching:** Retrieves stock price data using Yahoo Finance.
- **Feature Engineering:** Computes technical indicators with [pandas-ta](https://github.com/twopirllc/pandas-ta):
  - Moving Averages (MA-20, EMA-20)
  - MACD (12,26,9) & Signal
  - RSI (14)
  - Volatility, Returns
- **Deep Learning Model:** LSTM-based neural network with dropout and dense layers.
- **Prediction Visualization:** Interactive charts for historical and predicted prices, volume, and confidence intervals.
- **Performance Metrics:** MAE, RMSE, MAPE, R².
- **Downloadable Results:** Export predictions as CSV.
- **Simple, Fast Pipeline:** Designed for quick experimentation and educational use.

## Architecture
The system architecture follows a streamlined workflow:

1. **Data Collection Layer**: Yahoo Finance API retrieves historical stock data
2. **Preprocessing Layer**: Technical indicators are calculated and data is normalized
3. **Model Layer**: LSTM neural network trained on prepared features
4. **Prediction Layer**: Future price predictions with confidence intervals
5. **Visualization Layer**: Interactive Streamlit UI displays results and insights

The application is built using a client-server architecture:
- **API Backend**: FastAPI service that handles data fetching, model training, and predictions
- **Frontend**: Streamlit application that provides the user interface

The LSTM model architecture consists of:
- Input layer with time-series features
- LSTM layer with 64 units
- Dropout layer (0.2) for regularization
- Dense layer with 32 units and ReLU activation
- Output layer for price prediction

![](/assets/architecture.png) 

## Project Structure

```
.
├── api/                # API backend service
│   ├── main.py         # FastAPI endpoints
│   └── data_cache/     # Cache for stock data
├── frontend/           # Streamlit frontend
│   └── main.py         # Streamlit UI application
├── model.py            # Feature engineering, model training, prediction, evaluation
├── main.py             # Standalone application (single-file version)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition for Docker/Podman
├── .gitignore          # Git ignore file
├── assets/             # Diagrams and images
│   └── architecture.png  # System architecture diagram
└── README.md           # Project documentation
```

## How It Works

1. **User selects a stock, date range, and prediction settings in the sidebar.**
2. **App fetches historical price data and computes technical indicators.**
3. **LSTM model is trained on the features with early stopping.**
4. **Future prices are predicted and visualized with performance metrics.**
5. **Results can be downloaded for further analysis.**

## Setup and Running the Application

### Option 1: Running locally with Python

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the API backend:**
    ```bash
    cd /path/to/lstm-stock
    uvicorn api.main:app --reload
    ```

3. **Run the Streamlit frontend in a separate terminal:**
    ```bash
    cd /path/to/lstm-stock
    streamlit run frontend/main.py
    ```

4. **Access the application:**
   - Frontend: http://localhost:8501
   - API docs: http://localhost:8000/docs

### Option 2: Running with Podman (Recommended)

1. **Build the container image:**
   ```bash
   cd /path/to/lstm-stock
   podman build -t lstm-stock .
   ```

2. **Run the API backend:**
   ```bash
   podman run -d -p 8000:8000 --name lstm-api lstm-stock
   ```

3. **Run the Streamlit frontend:**
   ```bash
   podman run -p 8501:8501 --name lstm-frontend lstm-stock streamlit run frontend/main.py --server.port 8501 --server.address 0.0.0.0
   ```

4. **Note about API connectivity:**
   - When running the frontend in a container, you may need to update the API_BASE_URL in frontend/main.py to point to your host machine's IP address instead of localhost.
   - For example: `API_BASE_URL = "http://192.168.1.100:8000"` (replace with your actual IP)

### Testing the API Directly

You can test the API endpoints directly using the Swagger UI:

1. **Open the API documentation:**
   - Go to http://localhost:8000/docs in your browser

2. **Test the /predict endpoint:**
   - Click on the POST /predict endpoint
   - Click "Try it out"
   - Use this JSON request body:
   ```json
   {
     "ticker": "AAPL",
     "start_date": "2022-05-01",
     "end_date": "2023-05-01",
     "prediction_days": 30,
     "time_steps": 60
   }
   ```
   - Click "Execute" to see the prediction results

3. **Test the /stock/{ticker} endpoint:**
   - Click on the GET /stock/{ticker} endpoint
   - Click "Try it out"
   - Enter a ticker symbol (e.g., "AAPL")
   - Click "Execute" to see detailed stock information

## Troubleshooting

### Connection Issues
- If the frontend can't connect to the API when running in containers, check that the API_BASE_URL is set correctly
- For local network access, use your machine's IP address instead of localhost

### Container Issues
- If Podman containers fail to start, check for port conflicts using `netstat -tuln`
- Ensure the ports 8000 and 8501 are available on your system

### Model Training Issues
- If you encounter memory errors during model training, try reducing the time_steps parameter
- For faster predictions, use a shorter date range or prediction_days value

## Notes

- The pipeline uses a simple LSTM model and a small set of technical indicators for speed and clarity.
- For production or research, you can extend the feature set, tune hyperparameters, or add more advanced models.
- The app is for educational and research purposes only.
- Supports international stocks with appropriate currency display (USD, INR, EUR, etc.)
- Now supports containerized deployment using Podman or Docker

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies
- Alternatively, use the provided Dockerfile to run in a container
- When using containers: Podman or Docker installed on your system

## Credits

- Built with [Streamlit](https://streamlit.io/), [TensorFlow/Keras](https://www.tensorflow.org/), [pandas-ta](https://github.com/twopirllc/pandas-ta), [yfinance](https://github.com/ranaroussi/yfinance), and [Plotly](https://plotly.com/python/).
- Backend API powered by [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/).
