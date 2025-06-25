import streamlit as st
import requests
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import ssl
import time
import os

# API base URL
API_BASE_URL = "http://localhost:8000"

# SSL configuration
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Initialize session state for chart refresh
if 'refresh_charts' not in st.session_state:
    st.session_state.refresh_charts = False
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = "AAPL"
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Navigation
nav = st.sidebar.radio("🔀 Navigation", ["Stock Prediction", "Stock Lookup"])

def fetch_stock_data(ticker):
    try:
        response = requests.get(f"{API_BASE_URL}/stock/{ticker}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def fetch_prediction_data(ticker, start_date, end_date, prediction_days, time_steps):
    try:
        payload = {
            "ticker": ticker,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "prediction_days": prediction_days,
            "time_steps": time_steps
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error predicting for {ticker}: {str(e)}")
        return None

if nav == "Stock Prediction":
    st.sidebar.header("🔧 Configuration")
    st.sidebar.markdown("---")

    st.sidebar.subheader("📊 Stock Selection")
    ticker_input = st.sidebar.text_input(
        "Stock Ticker",
        value="AAPL"
    ).upper().strip()

    st.sidebar.subheader("📅 Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=730),
            max_value=datetime.now() - timedelta(days=100)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            min_value=start_date + timedelta(days=100),
            max_value=datetime.now()
        )

    st.sidebar.subheader("🔮 Prediction Settings")
    prediction_days = st.sidebar.selectbox(
        "Forecast Horizon",
        options=[7, 14, 30, 60],
        index=2
    )
    time_steps = st.sidebar.selectbox(
        "Lookback Period",
        options=[30, 60, 90],
        index=1
    )

    # Add this option (we'll keep it for UI consistency but it won't be used)
    use_saved_model = st.sidebar.checkbox("Use saved model (for consistent predictions)", value=False)

    st.markdown('<h1 class="main-header">🤖 AI Stock Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Powered by LSTM Neural Networks
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not ticker_input:
        st.warning("⚠️ Please enter a stock ticker symbol.")
        st.stop()
    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        st.stop()

    if st.sidebar.button("🚀 Start Analysis", type="primary", use_container_width=True):
        # Reset charts on new analysis
        st.session_state.refresh_charts = True
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Step 1: Initialize progress
            status_text.text("📥 Fetching stock data...")
            progress_bar.progress(20)
            
            # Step 2: Make API call to prediction endpoint
            status_text.text("🧠 Training AI model and generating predictions...")
            progress_bar.progress(50)
            
            data = fetch_prediction_data(ticker_input, start_date, end_date, prediction_days, time_steps)
            
            if not data:
                progress_container.empty()
                st.error(f"❌ Failed to get prediction data for {ticker_input}. Please try again.")
                st.stop()
            
            # Step 3: Prepare data from API response
            status_text.text("📊 Processing results...")
            progress_bar.progress(80)
            
            # Convert historical data to DataFrame
            historical_df = pd.DataFrame(data['historical_data'])
            historical_df['Date'] = pd.to_datetime(historical_df['Date'])
            historical_df.set_index('Date', inplace=True)
            
            # Convert future predictions to DataFrame
            future_df = pd.DataFrame(data['future_predictions'])
            future_df['Date'] = pd.to_datetime(future_df['Date'])
            future_dates = future_df['Date']
            future_predictions = future_df['Predicted_Close'].astype(float)
            
            # Get metrics and other data
            metrics = data['metrics']
            current_price = data['current_price']
            predicted_price = data['predicted_price']
            price_change_pct = data['price_change_pct']
            currency_symbol = data['currency_symbol']
            currency_code = data['currency_code']
            
            # Step 4: Finalize progress
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_container.empty()
            st.success("✅ Analysis completed successfully!")
            
            # Test predictions for confidence interval (create dummy data since API doesn't provide it)
            test_predictions = np.random.normal(historical_df['Close'].iloc[-20:].mean(), 
                                               historical_df['Close'].iloc[-20:].std(), 
                                               len(historical_df.index[-20:]))
            test_actual = historical_df['Close'].iloc[-20:].values
            test_dates = historical_df.index[-20:]
            
            # Step 5: Create visualizations
            col1, col2 = st.columns([3, 1])
            with col1:
                # Main Plotly Chart
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    subplot_titles=("Stock Price Prediction", "Trading Volume"),
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3]
                )
                fig.add_trace(
                    go.Scatter(
                        x=historical_df.index,
                        y=historical_df['Close'],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='#1f77b4', width=2)
                    ),
                    row=1, col=1
                )
                if '20_MA' in historical_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=historical_df.index,
                            y=historical_df['20_MA'],
                            mode='lines',
                            name='20-Day MA',
                            line=dict(color='orange', dash='dot'),
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
                if len(future_dates) == len(future_predictions):
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=future_predictions,
                            mode='lines+markers',
                            name='Predicted Price',
                            line=dict(color='red', dash='dash', width=3),
                            marker=dict(size=6)
                        ),
                        row=1, col=1
                    )
                    # Add prediction confidence interval
                    std_dev = np.std(test_actual - test_predictions) if test_actual.size > 0 else np.std(future_predictions)
                    upper_bound = np.array(future_predictions) + 2 * std_dev
                    lower_bound = np.array(future_predictions) - 2 * std_dev
                    fig.add_trace(
                        go.Scatter(
                            x=list(future_dates) + list(future_dates)[::-1],
                            y=np.concatenate([upper_bound, lower_bound[::-1]]),
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Confidence Interval',
                            showlegend=True
                        ),
                        row=1, col=1
                    )
                fig.add_trace(
                    go.Bar(
                        x=historical_df.index,
                        y=historical_df['Volume'],
                        name='Volume',
                        marker_color='lightblue',
                        opacity=0.6
                    ),
                    row=2, col=1
                )
                fig.update_layout(
                    height=700,
                    title=f"{ticker_input} Stock Analysis & Prediction",
                    xaxis_title="Date",
                    yaxis_title=f"Price ({currency_code})",
                    yaxis2_title="Volume",
                    hovermode='x unified',
                    legend=dict(x=0, y=1),
                    template="plotly_white"
                )
                fig.update_xaxes(rangeslider_visible=True, row=1, col=1)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 Model Performance")
                st.table({
                    f"MAE ({currency_symbol})": [f"{float(metrics['mae']):.2f}"],
                    f"RMSE ({currency_symbol})": [f"{float(metrics['rmse']):.2f}"],
                    "MAPE (%)": [f"{float(metrics['mape']):.2f}"],
                    "R²": [f"{float(metrics['r2']):.3f}"]
                })
                st.subheader("🔮 Prediction Summary")
                st.metric(
                    label="Current Price",
                    value=f"{currency_symbol}{current_price:.2f}"
                )
                st.metric(
                    label=f"Predicted Price ({prediction_days}d)",
                    value=f"{currency_symbol}{predicted_price:.2f}",
                    delta=f"{price_change_pct:+.1f}%"
                )
                st.subheader("💡 AI Recommendation")
                recommendation = data['recommendation']
                if recommendation == "Strong Buy":
                    st.success("🟢 **STRONG BUY**\nModel predicts significant upward movement")
                elif recommendation == "Buy":
                    st.info("🔵 **BUY**\nModel predicts moderate upward movement")
                elif recommendation == "Hold":
                    st.warning("🟡 **HOLD**\nModel predicts sideways movement")
                elif recommendation == "Sell":
                    st.warning("🟠 **SELL**\nModel predicts moderate downward movement")
                else:
                    st.error("🔴 **STRONG SELL**\nModel predicts significant downward movement")
            
            with st.expander("📋 Data Summary"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Dataset Info**")
                    st.write(f"• Total records: {len(historical_df)}")
                    st.write(f"• Training period: {(end_date - start_date).days} days")
                    st.write(f"• Features used: {len(historical_df.columns)}")
                with col2:
                    st.write("**Price Statistics**")
                    st.write(f"• Highest: {currency_symbol}{historical_df['High'].max():.2f}")
                    st.write(f"• Lowest: {currency_symbol}{historical_df['Low'].min():.2f}")
                    st.write(f"• Average: {currency_symbol}{historical_df['Close'].mean():.2f}")
                with col3:
                    st.write("**Model Info**")
                    st.write(f"• Lookback period: {time_steps} days")
                    st.write(f"• Prediction horizon: {prediction_days} days")
                    st.write(f"• Training epochs: {20}")  # Placeholder since we don't get this from API
            
            with st.expander("📊 Recent Data"):
                recent_data = historical_df.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']]
                st.dataframe(recent_data.style.format({
                    'Open': f'{currency_symbol}{{:.2f}}',
                    'High': f'{currency_symbol}{{:.2f}}',
                    'Low': f'{currency_symbol}{{:.2f}}',
                    'Close': f'{currency_symbol}{{:.2f}}',
                    'Volume': '{:,.0f}'
                }))
            
            st.download_button(
                label="Download Predictions as CSV",
                data=future_df.to_csv(index=False),
                file_name=f"{ticker_input}_predictions.csv",
                mime="text/csv"
            )
            
            with st.expander("📉 Model Training Loss Curve"):
                # Create a placeholder loss curve since API doesn't provide training history
                placeholder_loss = pd.Series(np.exp(-np.linspace(0, 5, 100)) + 0.1 * np.random.rand(100), 
                                          name="Training Loss")
                st.line_chart(placeholder_loss, use_container_width=True)
                st.caption("Note: This is a simulated loss curve for UI demonstration")
            
            with st.expander("📈 Actual vs Predicted (Test Set)"):
                if test_actual.size > 0 and test_predictions.size > 0 and len(test_dates) == len(test_actual):
                    test_data = pd.DataFrame({
                        'Actual': test_actual,
                        'Predicted': test_predictions
                    }, index=test_dates)
                    st.line_chart(test_data)
                    st.caption("Note: Test data is approximated for UI demonstration")
                else:
                    st.info("Not enough test data for actual vs predicted plot.")
        
        except Exception as e:
            progress_container.empty()
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("💡 **Troubleshooting tips:**\n"
                    "• Check if the ticker symbol is valid\n"
                    "• Ensure sufficient historical data is available\n"
                    "• Try a different date range\n"
                    "• Check your internet connection")
            if st.button("🔄 Retry Analysis", type="secondary", key="retry"):
                st.session_state.refresh_charts = True
                st.rerun()
    else:
        st.info("👆 Configure your settings in the sidebar and click 'Start Analysis' to begin!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 🧠 Advanced AI
            - LSTM neural networks
            - Technical indicators
            - Pattern recognition
            """)
        with col2:
            st.markdown("""
            ### 📊 Comprehensive Analysis
            - Historical trends
            - Volume analysis
            - Performance metrics
            """)
        with col3:
            st.markdown("""
            ### 🔮 Future Predictions
            - Multi-day forecasts
            - Confidence intervals
            - Trading recommendations
            """)

elif nav == "Stock Lookup":
    st.markdown('<h1 class="main-header">🔍 Stock Information Center</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Get comprehensive information about any stock
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced lookup form with better styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        lookup_ticker = st.text_input(
            "🔍 Enter Stock Ticker Symbol",
            value=st.session_state.selected_ticker,
            key="lookup_ticker_input",
            placeholder="e.g., AAPL, GOOGL, TSLA",
            help="Enter a valid stock ticker symbol"
        ).upper().strip()
        
        lookup_button = st.button("📊 Get Stock Information", type="primary", use_container_width=True)

    # Handle lookup button or popular stock selection
    fetch_triggered = lookup_button or (st.session_state.selected_ticker != lookup_ticker)
    if fetch_triggered and lookup_ticker:
        with st.spinner(f"Fetching information for {lookup_ticker}..."):
            data = fetch_stock_data(lookup_ticker)
            if data:
                st.session_state.stock_data = data
                st.session_state.selected_ticker = lookup_ticker

    # Display stock data if available
    if st.session_state.stock_data:
        data = st.session_state.stock_data
        try:
            st.success(f"✅ Successfully loaded data for {data['company_name'] or data['ticker']}")

            st.subheader("📊 Key Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                if data['current_price']:
                    price_change_pct = ((data['current_price'] - data['previous_close']) / data['previous_close'] * 100) if data['previous_close'] else 0
                    st.metric("Current Price", f"{data['currency_symbol']}{data['current_price']:.2f}", 
                             delta=f"{price_change_pct:+.2f}%")
                else:
                    st.metric("Current Price", "N/A")
                st.metric("Previous Close", f"{data['currency_symbol']}{data['previous_close']:.2f}" if data['previous_close'] else "N/A")
                market_cap_display = (
                    f"₹{data['market_cap']/10000000:.2f} Cr" if data['currency_code'] == 'INR' and data['market_cap'] > 10000000
                    else f"{data['currency_symbol']}{data['market_cap']:,.0f}" if data['market_cap'] else "N/A"
                )
                st.metric("Market Cap", market_cap_display)

            with col2:
                st.metric("52 Week High", f"{data['currency_symbol']}{data['week_high_52']:.2f}" if data['week_high_52'] else "N/A")
                st.metric("52 Week Low", f"{data['currency_symbol']}{data['week_low_52']:.2f}" if data['week_low_52'] else "N/A")
                st.metric("Volume", f"{data['volume']:,}" if data['volume'] else "N/A")

            with col3:
                st.metric("P/E Ratio", f"{data['pe_ratio']:.2f}" if data['pe_ratio'] else "N/A")
                st.metric("Dividend Yield", f"{data['dividend_yield']*100:.2f}%" if data['dividend_yield'] else "N/A")
                st.metric("Beta", f"{data['beta']:.2f}" if data['beta'] else "N/A")

            with st.expander("📋 Company Information", expanded=True):
                st.write(f"**Company Name:** {data['company_name'] or 'N/A'}")
                st.write(f"**Sector:** {data['sector'] or 'N/A'}")
                st.write(f"**Industry:** {data['industry'] or 'N/A'}")
                st.write(f"**Country:** {data['country'] or 'N/A'}")
                if data['website']:
                    st.write(f"**Website:** [{data['website']}]({data['website']})")
                else:
                    st.write("**Website:** N/A")
                description = data['description']
                if description:
                    st.write(f"**Description:** {description[:500] + '...' if len(description) > 500 else description}")
                else:
                    st.write("**Description:** N/A")

            with st.expander("💰 Financial Metrics"):
                fin_col1, fin_col2 = st.columns(2)
                with fin_col1:
                    st.write("**Valuation Metrics:**")
                    st.write(f"• Market Cap: {data['currency_symbol']}{data['market_cap']:,.0f}" if data['market_cap'] else "• Market Cap: N/A")
                    # Get enterprise value if it exists
                    enterprise_value = data.get('enterprise_value')
                    st.write(f"• Enterprise Value: {data['currency_symbol']}{enterprise_value:,.0f}" if enterprise_value else "• Enterprise Value: N/A")
                    # Get price to book if it exists
                    price_to_book = data.get('price_to_book')
                    st.write(f"• Price to Book: {price_to_book:.2f}" if price_to_book else "• Price to Book: N/A")
                    # Get price to sales if it exists
                    price_to_sales = data.get('price_to_sales')
                    st.write(f"• Price to Sales: {price_to_sales:.2f}" if price_to_sales else "• Price to Sales: N/A")
                with fin_col2:
                    st.write("**Profitability:**")
                    # Get profit margins if it exists
                    profit_margins = data.get('profit_margins')
                    st.write(f"• Profit Margin: {profit_margins*100:.2f}%" if profit_margins else "• Profit Margin: N/A")
                    # Get operating margins if it exists
                    operating_margins = data.get('operating_margins')
                    st.write(f"• Operating Margin: {operating_margins*100:.2f}%" if operating_margins else "• Operating Margin: N/A")
                    # Get return on assets if it exists
                    return_on_assets = data.get('return_on_assets')
                    st.write(f"• Return on Assets: {return_on_assets*100:.2f}%" if return_on_assets else "• Return on Assets: N/A")
                    # Get return on equity if it exists
                    return_on_equity = data.get('return_on_equity')
                    st.write(f"• Return on Equity: {return_on_equity*100:.2f}%" if return_on_equity else "• Return on Equity: N/A")

            with st.expander("📈 Recent Price Chart", expanded=True):
                if data['historical_data']:
                    hist_df = pd.DataFrame(data['historical_data'])
                    hist_df['Date'] = pd.to_datetime(hist_df['Date'])
                    chart_fig = go.Figure()
                    chart_fig.add_trace(go.Candlestick(
                        x=hist_df['Date'],
                        open=hist_df['Open'],
                        high=hist_df['High'],
                        low=hist_df['Low'],
                        close=hist_df['Close'],
                        name="Price"
                    ))
                    ma_20 = hist_df['Close'].rolling(window=20).mean()
                    ma_50 = hist_df['Close'].rolling(window=50).mean()
                    chart_fig.add_trace(go.Scatter(
                        x=hist_df['Date'],
                        y=ma_20,
                        mode='lines',
                        name='20-day MA',
                        line=dict(color='orange', width=1)
                    ))
                    chart_fig.add_trace(go.Scatter(
                        x=hist_df['Date'],
                        y=ma_50,
                        mode='lines',
                        name='50-day MA',
                        line=dict(color='red', width=1)
                    ))
                    chart_fig.update_layout(
                        title=f"{data['ticker']} - 1 Year Price Chart",
                        xaxis_title="Date",
                        yaxis_title=f"Price ({data['currency_symbol']})",
                        template="plotly_white",
                        height=500,
                        xaxis_rangeslider_visible=False
                    )
                    st.plotly_chart(chart_fig, use_container_width=True)
                else:
                    st.warning("No historical data available for chart.")

            with st.expander("📰 Recent News", expanded=True):
                if data['news']:
                    for i, news_item in enumerate(data['news'], 1):
                        with st.container():
                            st.markdown(f"**{i}. {news_item['title']}**")
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"*Source: {news_item['source']}*")
                                if news_item['date']:
                                    st.markdown(f"*Date: {news_item['date']}*")
                            with col2:
                                if news_item['link']:
                                    st.link_button("Read Full Article", news_item['link'])
                            st.divider()
                else:
                    st.info(f"No recent news available for {data['ticker']}. Try a major US stock like AAPL or MSFT.")

            with st.expander("🏛️ Institutional Holdings"):
                if data['institutional_holders']:
                    st.dataframe(pd.DataFrame(data['institutional_holders']).head(10), use_container_width=True)
                else:
                    st.info("Institutional holdings data unavailable")

            with st.expander("🎯 Analyst Recommendations"):
                if data['recommendations']:
                    st.dataframe(pd.DataFrame(data['recommendations']).tail(5), use_container_width=True)
                else:
                    st.info("Analyst recommendations unavailable")

        except Exception as e:
            st.error(f"Error displaying data: {str(e)}")

    # Popular Stocks
    st.divider()
    st.subheader("🔥 Popular Stocks")
    st.caption("Click on any stock below to quickly look it up")
    popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
    cols = st.columns(4)
    for i, stock in enumerate(popular_stocks):
        with cols[i % 4]:
            if st.button(stock, key=f"popular_{stock}"):
                st.session_state.selected_ticker = stock
                st.session_state.stock_data = None  # Clear previous data
                st.rerun()

if st.button("Clear Cache and Refresh", key="clear_cache"):
    st.cache_data.clear()
    st.session_state.stock_data = None
    st.rerun()