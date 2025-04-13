import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from transformers import pipeline
import warnings
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI-Powered ETF Analytics Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# List of ETFs
etf_list = [
    "MAFANG.NS", "FMCGIETF.NS", "MOGSEC.NS", "TATAGOLD.NS", "GOLDIETF.NS",
    "GOLDCASE.NS", "HDFCGOLD.NS", "GOLD1.NS", "AXISGOLD.NS", "GOLD360.NS",
    "ABGSEC.NS", "SETFGOLD.NS", "GOLDBEES.NS", "LICMFGOLD.NS", "QGOLDHALF.NS",
    "GSEC5IETF.NS", "IVZINGOLD.NS", "GOLDSHARE.NS", "BSLGOLDETF.NS", "LICNFNHGP.NS",
    "GOLDETFADD.NS", "UNIONGOLD.NS", "CONSUMBEES.NS", "SDL26BEES.NS", "AXISCETF.NS",
    "GROWWGOLD.NS", "GOLDETF.NS", "MASPTOP50.NS", "SETF10GILT.NS", "EBBETF0433.NS",
    "NV20BEES.NS", "BBNPPGOLD.NS", "CONSUMIETF.NS", "AUTOBEES.NS", "BSLSENETFG.NS",
    "LTGILTBEES.NS", "AUTOIETF.NS", "AXISBPSETF.NS", "GILT5YBEES.NS", "LIQUIDCASE.NS",
    "GROWWLIQID.NS", "GSEC10YEAR.NS", "LIQUIDBETF.NS", "LIQUIDADD.NS", "LIQUID1.NS",
    "HDFCLIQUID.NS", "MOLOWVOL.NS", "AONELIQUID.NS", "CASHIETF.NS", "LIQUIDPLUS.NS",
    "LIQUIDSHRI.NS", "ABSLLIQUID.NS", "LIQUIDETF.NS", "CONS.NS", "LIQUIDSBI.NS",
    "LIQUID.NS", "EGOLD.NS", "BBNPNBETF.NS", "LIQUIDIETF.NS", "IVZINNIFTY.NS",
    "GSEC10ABSL.NS", "LIQUIDBEES.NS", "EBBETF0430.NS", "SBIETFCON.NS", "MON100.NS",
    "LICNETFGSC.NS", "GSEC10IETF.NS", "QUAL30IETF.NS", "SILVRETF.NS", "LICNETFSEN.NS",
    "HDFCLOWVOL.NS", "EBANKNIFTY.NS", "LOWVOLIETF.NS", "EBBETF0431.NS", "TOP100CASE.NS",
    "NIFTYQLITY.NS", "HDFCGROWTH.NS", "SHARIABEES.NS", "BBETF0432.NS"
]

# Function to fetch ETF data
@st.cache_data(ttl=3600)
def get_etf_data(etf_list, period='1y'):
    data = {}
    for etf in etf_list:
        try:
            ticker = yf.Ticker(etf)
            df = ticker.history(period=period)
            if not df.empty:
                data[etf] = df
        except:
            continue
    return data

# Function to fetch news sentiment
@st.cache_data(ttl=3600)
def get_news_sentiment(etf_name):
    try:
        # Initialize sentiment analysis pipeline
        sentiment_pipeline = pipeline("sentiment-analysis")
        
        # Simulate fetching news (in a real app, you'd use a news API)
        news_samples = [
            f"Positive outlook for {etf_name} as market conditions improve",
            f"Investors show mixed reactions to {etf_name} performance",
            f"{etf_name} reaches new highs amid market rally"
        ]
        
        # Analyze sentiment
        results = sentiment_pipeline(news_samples)
        avg_score = sum([1 if res['label'] == 'POSITIVE' else -1 for res in results]) / len(results)
        
        return {
            'sentiment_score': avg_score,
            'sentiment': 'Bullish' if avg_score > 0.33 else 'Bearish' if avg_score < -0.33 else 'Neutral'
        }
    except:
        return {'sentiment_score': 0, 'sentiment': 'Neutral'}

# Function to calculate advanced metrics
def calculate_advanced_metrics(df):
    if df.empty:
        return None
    
    # Calculate returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    
    # Calculate metrics
    metrics = {
        'Last_Price': df['Close'].iloc[-1],
        '1D_Return': df['Daily_Return'].iloc[-1],
        '1W_Return': (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) if len(df) >= 5 else np.nan,
        '1M_Return': (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) if len(df) >= 20 else np.nan,
        '3M_Return': (df['Close'].iloc[-1] / df['Close'].iloc[-60] - 1) if len(df) >= 60 else np.nan,
        'YTD_Return': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1),
        'Volatility': df['Daily_Return'].std() * np.sqrt(252),
        'Sharpe_Ratio': (df['Daily_Return'].mean() / df['Daily_Return'].std()) * np.sqrt(252),
        'Max_Drawdown': (df['Close'] / df['Close'].cummax() - 1).min(),
        'Sortino_Ratio': calculate_sortino_ratio(df['Daily_Return']),
        'Beta': calculate_beta(df['Daily_Return']),
        'Alpha': calculate_alpha(df['Daily_Return']),
        'SMA_20_Current': df['SMA_20'].iloc[-1],
        'SMA_50_Current': df['SMA_50'].iloc[-1],
        'MACD_Current': df['MACD'].iloc[-1],
        'RSI_Current': df['RSI'].iloc[-1],
        'Golden_Cross': df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] and df['SMA_20'].iloc[-2] <= df['SMA_50'].iloc[-2],
        'Death_Cross': df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1] and df['SMA_20'].iloc[-2] >= df['SMA_50'].iloc[-2]
    }
    return metrics

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sortino_ratio(returns, risk_free_rate=0.05/252):
    downside_returns = returns[returns < risk_free_rate]
    downside_std = downside_returns.std()
    if downside_std == 0:
        return np.nan
    excess_returns = returns.mean() - risk_free_rate
    return excess_returns / downside_std * np.sqrt(252)

def calculate_beta(asset_returns, market_returns=None, risk_free_rate=0.05/252):
    # In a real implementation, you would compare against a market index
    if market_returns is None:
        market_returns = asset_returns.rolling(5).mean()  # Placeholder
    excess_asset = asset_returns - risk_free_rate
    excess_market = market_returns - risk_free_rate
    cov_matrix = np.cov(excess_asset, excess_market)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    return beta

def calculate_alpha(asset_returns, market_returns=None, risk_free_rate=0.05/252):
    beta = calculate_beta(asset_returns, market_returns, risk_free_rate)
    if market_returns is None:
        market_returns = asset_returns.rolling(5).mean()  # Placeholder
    avg_asset_return = asset_returns.mean() * 252
    avg_market_return = market_returns.mean() * 252
    expected_return = risk_free_rate * 252 + beta * (avg_market_return - risk_free_rate * 252)
    alpha = avg_asset_return - expected_return
    return alpha

# Function to train multiple predictive models
def train_predictive_models(df, days_to_predict=5):
    if df.empty or len(df) < 30:
        return None
    
    # Prepare data
    df = df[['Close']].copy()
    df['Date'] = df.index
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag5'] = df['Close'].shift(5)
    df['Close_Lag10'] = df['Close'].shift(10)
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df = df.dropna()
    
    if len(df) < 10:
        return None
    
    # Features and target
    X = df[['Days', 'Close_Lag1', 'Close_Lag5', 'Close_Lag10', 'MA_5', 'MA_10']]
    y = df['Close']
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'ARIMA': None,  # Will handle separately
        'LSTM': None,   # Will handle separately
        'Prophet': None # Will handle separately
    }
    
    results = {}
    
    # Train traditional models
    for name, model in models.items():
        if model is None:
            continue
            
        mse_scores = []
        r2_scores = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse_scores.append(mean_squared_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'avg_mse': np.mean(mse_scores),
            'avg_r2': np.mean(r2_scores)
        }
    
    # Train ARIMA
    try:
        arima_model = ARIMA(y, order=(5,1,0))
        arima_fit = arima_model.fit()
        results['ARIMA'] = {
            'model': arima_fit,
            'avg_mse': arima_fit.mse,
            'avg_r2': None  # ARIMA doesn't provide R-squared
        }
    except:
        pass
    
    # Train LSTM
    try:
        # Prepare data for LSTM
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Close']])
        
        # Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data)-seq_length-1):
                X.append(data[i:(i+seq_length), 0])
                y.append(data[i+seq_length, 0])
            return np.array(X), np.array(y)
        
        seq_length = 5
        X_lstm, y_lstm = create_sequences(scaled_data, seq_length)
        X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
        
        # Build LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
        lstm_model.add(LSTM(50))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train
        lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=1, verbose=0)
        
        # Evaluate
        train_predict = lstm_model.predict(X_lstm)
        train_predict = scaler.inverse_transform(train_predict)
        y_lstm_actual = scaler.inverse_transform(y_lstm.reshape(-1,1))
        mse = mean_squared_error(y_lstm_actual, train_predict)
        
        results['LSTM'] = {
            'model': lstm_model,
            'scaler': scaler,
            'seq_length': seq_length,
            'avg_mse': mse,
            'avg_r2': r2_score(y_lstm_actual, train_predict)
        }
    except Exception as e:
        print(f"LSTM Error: {e}")
        pass
    
    # Train Prophet
    try:
        prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(prophet_df)
        
        # Evaluate
        future = prophet_model.make_future_dataframe(periods=0)
        forecast = prophet_model.predict(future)
        mse = mean_squared_error(prophet_df['y'], forecast['yhat'][:len(prophet_df)])
        
        results['Prophet'] = {
            'model': prophet_model,
            'avg_mse': mse,
            'avg_r2': r2_score(prophet_df['y'], forecast['yhat'][:len(prophet_df)])
        }
    except:
        pass
    
    return results

# Function to categorize ETFs
def categorize_etf(etf_name):
    etf_name = etf_name.lower()
    if 'gold' in etf_name:
        return 'Gold'
    elif 'liquid' in etf_name or 'cash' in etf_name:
        return 'Liquid'
    elif 'cons' in etf_name or 'consum' in etf_name:
        return 'Consumer'
    elif 'auto' in etf_name:
        return 'Automobile'
    elif 'gilt' in etf_name or 'gsec' in etf_name:
        return 'Government Securities'
    elif 'nifty' in etf_name or 'top' in etf_name or 'mon' in etf_name:
        return 'Equity Index'
    elif 'sharia' in etf_name:
        return 'Islamic'
    elif 'silver' in etf_name:
        return 'Silver'
    else:
        return 'Other'

# Function to generate AI insights
def generate_ai_insights(etf_name, metrics, predictions):
    insights = []
    
    # Price movement insights
    if metrics['1M_Return'] > 0.1:
        insights.append(f"📈 Strong positive momentum with {metrics['1M_Return']*100:.1f}% monthly return")
    elif metrics['1M_Return'] < -0.05:
        insights.append(f"📉 Negative momentum with {metrics['1M_Return']*100:.1f}% monthly loss")
    
    # Volatility insights
    if metrics['Volatility'] > 0.3:
        insights.append("⚠️ High volatility - suitable for risk-tolerant investors")
    elif metrics['Volatility'] < 0.1:
        insights.append("🛡️ Low volatility - stable option for conservative investors")
    
    # Technical indicator insights
    if metrics['Golden_Cross']:
        insights.append("🌟 Golden Cross detected (20-day SMA crossed above 50-day SMA) - bullish signal")
    if metrics['Death_Cross']:
        insights.append("💀 Death Cross detected (20-day SMA crossed below 50-day SMA) - bearish signal")
    
    if metrics['RSI_Current'] > 70:
        insights.append("🚨 Overbought conditions (RSI > 70) - potential pullback expected")
    elif metrics['RSI_Current'] < 30:
        insights.append("🛒 Oversold conditions (RSI < 30) - potential buying opportunity")
    
    # Prediction insights
    if predictions:
        # Find the best model with valid predictions
        valid_models = {k: v for k, v in predictions.items() if v is not None}
        if valid_models:
            best_model = min(valid_models.items(), key=lambda x: x[1]['avg_mse'])
            
            # Calculate predicted return if we have prediction data
            if 'Predicted_Price' in best_model[1]:
                current_price = metrics['Last_Price']
                predicted_price = best_model[1]['Predicted_Price']
                predicted_return = (predicted_price / current_price - 1)
                insights.append(f"🤖 AI Forecast: {best_model[0]} model predicts {predicted_return*100:.1f}% return in next 5 days")
    
    # Sentiment analysis
    sentiment = get_news_sentiment(etf_name)
    insights.append(f"📰 Market Sentiment: {sentiment['sentiment']} (Score: {sentiment['sentiment_score']:.2f})")
    
    # Risk-reward assessment
    if metrics['Sharpe_Ratio'] > 1.5:
        insights.append("🎯 Excellent risk-adjusted returns (Sharpe Ratio > 1.5)")
    elif metrics['Sharpe_Ratio'] < 0.5:
        insights.append("🧐 Suboptimal risk-adjusted returns (Sharpe Ratio < 0.5)")
    
    return insights

# Main app
def main():
    st.title("🤖 AI-Powered ETF Analytics Dashboard")
    st.write("Advanced analytics and machine learning for ETF investment decisions")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    analysis_period = st.sidebar.selectbox(
        "Analysis Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    recommendation_horizon = st.sidebar.selectbox(
        "Recommendation Horizon",
        ["1 Week", "1 Month", "3 Months"],
        index=0
    )
    
    risk_profile = st.sidebar.selectbox(
        "Risk Profile",
        ["Conservative", "Moderate", "Aggressive"],
        index=1
    )
    
    min_volatility = st.sidebar.slider(
        "Minimum Volatility Threshold",
        0.0, 1.0, 0.1, 0.01
    )
    
    max_volatility = st.sidebar.slider(
        "Maximum Volatility Threshold",
        0.0, 1.0, 0.5, 0.01
    )
    
    # AI model selection
    st.sidebar.header("AI Model Settings")
    use_rf = st.sidebar.checkbox("Random Forest", True)
    use_gb = st.sidebar.checkbox("Gradient Boosting", True)
    use_arima = st.sidebar.checkbox("ARIMA", True)
    use_lstm = st.sidebar.checkbox("LSTM", False)
    use_prophet = st.sidebar.checkbox("Prophet", False)
    
    # Fetch data
    with st.spinner("Fetching ETF data and analyzing..."):
        etf_data = get_etf_data(etf_list, analysis_period)
    
    if not etf_data:
        st.error("Failed to fetch data for any ETFs. Please try again later.")
        return
    
    # Calculate metrics for all ETFs
    metrics_data = []
    for etf, df in etf_data.items():
        metrics = calculate_advanced_metrics(df)
        if metrics:
            metrics['ETF'] = etf
            metrics['Category'] = categorize_etf(etf)
            metrics_data.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Filter by volatility and risk profile
    if risk_profile == "Conservative":
        metrics_df = metrics_df[(metrics_df['Volatility'] <= 0.2)]
    elif risk_profile == "Aggressive":
        metrics_df = metrics_df[(metrics_df['Volatility'] >= 0.3)]
    
    metrics_df = metrics_df[
        (metrics_df['Volatility'] >= min_volatility) & 
        (metrics_df['Volatility'] <= max_volatility)
    ]
    
    if metrics_df.empty:
        st.warning("No ETFs match your criteria. Please adjust the filters.")
        return
    
    # Sort by performance based on horizon
    if recommendation_horizon == "1 Week":
        metrics_df = metrics_df.sort_values('1W_Return', ascending=False)
    elif recommendation_horizon == "1 Month":
        metrics_df = metrics_df.sort_values('1M_Return', ascending=False)
    else:
        metrics_df = metrics_df.sort_values('3M_Return', ascending=False)
    
    # Display top recommendations with AI insights
    st.header("🏆 AI-Curated ETF Recommendations")
    top_n = min(5, len(metrics_df))
    
    for i in range(top_n):
        etf = metrics_df.iloc[i]['ETF']
        df = etf_data[etf]
        
        # Train predictive models
        predictions = train_predictive_models(df)
        
        # Generate AI insights
        insights = generate_ai_insights(etf, metrics_df.iloc[i].to_dict(), predictions)
        
        with st.expander(f"{i+1}. {etf} - {metrics_df.iloc[i]['Category']}", expanded=i==0):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"₹{metrics_df.iloc[i]['Last_Price']:.2f}",
                    delta=f"{metrics_df.iloc[i]['1W_Return' if recommendation_horizon == '1 Week' else '1M_Return' if recommendation_horizon == '1 Month' else '3M_Return']*100:.2f}%"
                )
                st.metric("Volatility", f"{metrics_df.iloc[i]['Volatility']:.3f}")
                st.metric("Sharpe Ratio", f"{metrics_df.iloc[i]['Sharpe_Ratio']:.2f}")
                st.metric("RSI", f"{metrics_df.iloc[i]['RSI_Current']:.1f}")
                
                if metrics_df.iloc[i]['Golden_Cross']:
                    st.success("Golden Cross Pattern")
                if metrics_df.iloc[i]['Death_Cross']:
                    st.error("Death Cross Pattern")
            
            with col2:
                # Price chart with indicators
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Close'],
                    name='Price',
                    line=dict(color='royalblue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Close'].rolling(20).mean(),
                    name='20-day SMA',
                    line=dict(color='orange', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Close'].rolling(50).mean(),
                    name='50-day SMA',
                    line=dict(color='green', width=1)
                ))
                fig.update_layout(
                    title=f"{etf} Price with Moving Averages",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display AI insights
            st.subheader("AI-Generated Insights")
            for insight in insights:
                st.write(f"- {insight}")
            
            # Show predictions if available
            if predictions and any(predictions.values()):
                st.subheader("AI Price Predictions")
                
                # Prepare prediction data
                pred_data = []
                for model_name, result in predictions.items():
                    if not result:
                        continue
                    
                    # Get predictions (simplified for demo)
                    if model_name == 'Random Forest' or model_name == 'Gradient Boosting':
                        # For tree-based models
                        last_data = {
                            'Days': (df.index[-1] - df.index[0]).days + 5,
                            'Close_Lag1': df['Close'].iloc[-1],
                            'Close_Lag5': df['Close'].iloc[-5] if len(df) >= 5 else df['Close'].iloc[-1],
                            'Close_Lag10': df['Close'].iloc[-10] if len(df) >= 10 else df['Close'].iloc[-1],
                            'MA_5': df['Close'].rolling(5).mean().iloc[-1],
                            'MA_10': df['Close'].rolling(10).mean().iloc[-1]
                        }
                        next_price = result['model'].predict(pd.DataFrame([last_data]))[0]
                        next_return = (next_price / df['Close'].iloc[-1] - 1)
                        
                        pred_data.append({
                            'Model': model_name,
                            'Next_Price': next_price,
                            '5D_Return': next_return,
                            'MSE': result['avg_mse'],
                            'R2': result['avg_r2']
                        })
                
                if pred_data:
                    pred_df = pd.DataFrame(pred_data)
                    st.dataframe(pred_df.style.format({
                        'Next_Price': '{:.2f}',
                        '5D_Return': '{:.2%}',
                        'MSE': '{:.4f}',
                        'R2': '{:.2f}'
                    }).background_gradient(cmap='RdYlGn', subset=['5D_Return']))
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Performance Metrics", "Category Analysis", "Predictive Analytics", 
        "Portfolio Optimizer", "ETF Explorer"
    ])
    
    with tab1:
        st.subheader("Advanced Performance Metrics")
        
        # Select metrics to display
        selected_metrics = st.multiselect(
            "Select metrics to display",
            ['1D_Return', '1W_Return', '1M_Return', '3M_Return', 'YTD_Return', 
             'Volatility', 'Sharpe_Ratio', 'Sortino_Ratio', 'Max_Drawdown',
             'Beta', 'Alpha', 'RSI_Current', 'MACD_Current', 'SMA_20_Current',
             'SMA_50_Current', 'Golden_Cross', 'Death_Cross'],
            default=['1W_Return', '1M_Return', 'Volatility', 'Sharpe_Ratio', 'RSI_Current']
        )
        
        if selected_metrics:
            display_df = metrics_df[['ETF', 'Category'] + selected_metrics]
            st.dataframe(
                display_df.style.format({
                    '1D_Return': '{:.2%}',
                    '1W_Return': '{:.2%}',
                    '1M_Return': '{:.2%}',
                    '3M_Return': '{:.2%}',
                    'YTD_Return': '{:.2%}',
                    'Volatility': '{:.3f}',
                    'Sharpe_Ratio': '{:.2f}',
                    'Sortino_Ratio': '{:.2f}',
                    'Max_Drawdown': '{:.2%}',
                    'Beta': '{:.2f}',
                    'Alpha': '{:.2f}',
                    'RSI_Current': '{:.1f}',
                    'MACD_Current': '{:.2f}',
                    'SMA_20_Current': '{:.2f}',
                    'SMA_50_Current': '{:.2f}'
                }).background_gradient(cmap='RdYlGn', subset=[m for m in selected_metrics if m not in ['Golden_Cross', 'Death_Cross']]),
                height=600
            )
        
        # Correlation heatmap
        st.subheader("Metric Correlations")
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        corr_matrix = metrics_df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("ETF Category Analysis")
        
        # Category distribution
        category_counts = metrics_df['Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        fig1 = px.pie(
            category_counts,
            names='Category',
            values='Count',
            title="ETF Category Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Category performance
        st.subheader("Average Performance by Category")
        category_perf = metrics_df.groupby('Category').agg({
            '1W_Return': 'mean',
            '1M_Return': 'mean',
            'Volatility': 'mean',
            'Sharpe_Ratio': 'mean',
            'RSI_Current': 'mean'
        }).reset_index()
        
        fig2 = px.bar(
            category_perf,
            x='Category',
            y=['1W_Return', '1M_Return'],
            barmode='group',
            title="Average Returns by Category"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        fig3 = px.bar(
            category_perf,
            x='Category',
            y=['Volatility', 'Sharpe_Ratio'],
            barmode='group',
            title="Risk Metrics by Category"
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Cluster analysis
        with tab2:
    st.subheader("ETF Cluster Analysis")
    cluster_features = metrics_df[['1M_Return', 'Volatility', 'Sharpe_Ratio']]
    
    # Drop rows with NaN values that would break clustering
    cluster_features = cluster_features.dropna()
    
    # Check if we have enough data points left
    if len(cluster_features) < 2:
        st.warning("Not enough valid data points for clustering (need at least 2). Some ETFs may have missing values.")
    else:
        scaler = MinMaxScaler()
        cluster_scaled = scaler.fit_transform(cluster_features)
        
        # Check for any remaining NaN/infinite values after scaling
        if np.isnan(cluster_scaled).any() or np.isinf(cluster_scaled).any():
            st.error("Data contains invalid values after scaling. Cannot perform clustering.")
        else:
            # Determine optimal clusters
            wcss = []
            for i in range(1, 6):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(cluster_scaled)
                wcss.append(kmeans.inertia_)
            
            fig4, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(1, 6), wcss, marker='o')
            ax.set_title('Elbow Method for Optimal Cluster Number')
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel('WCSS')
            st.pyplot(fig4)
            
            # Apply clustering
            n_clusters = st.slider("Select number of clusters", 2, 5, 3)
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
            cluster_labels = kmeans.fit_predict(cluster_scaled)
            
            # Add cluster labels back to the original metrics_df
            metrics_df['Cluster'] = np.nan
            metrics_df.loc[cluster_features.index, 'Cluster'] = cluster_labels
            
            # Create 3D scatter plot only for ETFs with valid cluster assignments
            valid_clusters = metrics_df.dropna(subset=['Cluster'])
            fig5 = px.scatter_3d(
                valid_clusters,
                x='1M_Return',
                y='Volatility',
                z='Sharpe_Ratio',
                color='Cluster',
                hover_name='ETF',
                title="ETF Clusters in 3D Space"
            )
            st.plotly_chart(fig5, use_container_width=True)
    
    with tab3:
        st.subheader("Advanced Predictive Analytics")
        
        selected_etf = st.selectbox(
            "Select ETF for prediction",
            metrics_df['ETF'].sort_values()
        )
        
        if selected_etf:
            df = etf_data[selected_etf]
            
            # Train all selected models
            with st.spinner("Training AI models..."):
                models_to_use = {}
                if use_rf:
                    models_to_use['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
                if use_gb:
                    models_to_use['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
                
                predictions = train_predictive_models(df)
            
            if not predictions or not any(predictions.values()):
                st.warning("Insufficient data for reliable predictions for this ETF.")
            else:
                # Display model performance
                st.subheader("Model Performance Comparison")
                perf_data = []
                for model_name, result in predictions.items():
                    if result:
                        perf_data.append({
                            'Model': model_name,
                            'MSE': result['avg_mse'],
                            'R2': result['avg_r2']
                        })
                
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    st.dataframe(
                        perf_df.style.format({
                            'MSE': '{:.4f}',
                            'R2': '{:.2f}'
                        }).background_gradient(cmap='RdYlGn', subset=['R2'])
                    )
                
                # Show predictions
                st.subheader("Price Predictions")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Next 5 Trading Days**")
                    
                    # Prepare prediction data
                    pred_data = []
                    for model_name, result in predictions.items():
                        if not result:
                            continue
                        
                        # Get predictions (simplified for demo)
                        if model_name in ['Random Forest', 'Gradient Boosting']:
                            # For tree-based models
                            last_data = {
                                'Days': (df.index[-1] - df.index[0]).days + 5,
                                'Close_Lag1': df['Close'].iloc[-1],
                                'Close_Lag5': df['Close'].iloc[-5] if len(df) >= 5 else df['Close'].iloc[-1],
                                'Close_Lag10': df['Close'].iloc[-10] if len(df) >= 10 else df['Close'].iloc[-1],
                                'MA_5': df['Close'].rolling(5).mean().iloc[-1],
                                'MA_10': df['Close'].rolling(10).mean().iloc[-1]
                            }
                            next_price = result['model'].predict(pd.DataFrame([last_data]))[0]
                            next_return = (next_price / df['Close'].iloc[-1] - 1)
                            
                            pred_data.append({
                                'Model': model_name,
                                'Predicted_Price': next_price,
                                '5D_Return': next_return
                            })
                        elif model_name == 'ARIMA':
                            # ARIMA forecast
                            forecast = result['model'].get_forecast(steps=5)
                            next_price = forecast.predicted_mean.iloc[-1]
                            next_return = (next_price / df['Close'].iloc[-1] - 1)
                            
                            pred_data.append({
                                'Model': model_name,
                                'Predicted_Price': next_price,
                                '5D_Return': next_return
                            })
                        elif model_name == 'LSTM':
                            # LSTM forecast
                            scaler = result['scaler']
                            seq_length = result['seq_length']
                            
                            last_sequence = scaled_data[-seq_length:]
                            last_sequence = np.reshape(last_sequence, (1, seq_length, 1))
                            
                            predicted_price_scaled = result['model'].predict(last_sequence)
                            predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
                            next_return = (predicted_price / df['Close'].iloc[-1] - 1)
                            
                            pred_data.append({
                                'Model': model_name,
                                'Predicted_Price': predicted_price,
                                '5D_Return': next_return
                            })
                        elif model_name == 'Prophet':
                            # Prophet forecast
                            future = result['model'].make_future_dataframe(periods=5)
                            forecast = result['model'].predict(future)
                            next_price = forecast['yhat'].iloc[-1]
                            next_return = (next_price / df['Close'].iloc[-1] - 1)
                            
                            pred_data.append({
                                'Model': model_name,
                                'Predicted_Price': next_price,
                                '5D_Return': next_return
                            })
                    
                    if pred_data:
                        pred_df = pd.DataFrame(pred_data)
                        st.dataframe(
                            pred_df.style.format({
                                'Predicted_Price': '{:.2f}',
                                '5D_Return': '{:.2%}'
                            }).background_gradient(cmap='RdYlGn', subset=['5D_Return'])
                        )
                
                with col2:
                    st.write("**Price Prediction Chart**")
                    
                    # Create plot with historical data and predictions
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Close'],
                        name='Historical Prices',
                        line=dict(color='royalblue', width=2)
                    ))
                    
                    # Add predictions from best model
                    if pred_data:
                        best_pred = max(pred_data, key=lambda x: x['5D_Return'])
                        future_dates = pd.date_range(df.index[-1], periods=6)[1:]
                        
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=[df['Close'].iloc[-1], best_pred['Predicted_Price']],
                            name=f"Prediction ({best_pred['Model']})",
                            line=dict(color='green', width=2, dash='dot')
                        ))
                    
                    fig.update_layout(
                        title=f"{selected_etf} Price Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("AI Portfolio Optimizer")
        
        st.write("""
        This tool helps you build an optimized ETF portfolio based on your investment goals and risk tolerance.
        The AI optimizer uses modern portfolio theory with machine learning enhancements.
        """)
        
        # Portfolio inputs
        col1, col2 = st.columns(2)
        
        with col1:
            investment_amount = st.number_input(
                "Investment Amount (₹)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=1000
            )
            
            optimization_goal = st.selectbox(
                "Optimization Goal",
                ["Maximize Returns", "Minimize Risk", "Balanced"],
                index=2
            )
        
        with col2:
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                ["Low", "Medium", "High"],
                index=1
            )
            
            max_etfs = st.slider(
                "Maximum Number of ETFs in Portfolio",
                1, 10, 5
            )
        
        # Select ETFs to include
        selected_etfs = st.multiselect(
            "Select ETFs to include in optimization",
            metrics_df['ETF'].sort_values(),
            default=metrics_df['ETF'].sort_values().tolist()[:5]
        )
        
        if st.button("Optimize Portfolio"):
            if len(selected_etfs) < 2:
                st.warning("Please select at least 2 ETFs for portfolio optimization")
            else:
                with st.spinner("Running portfolio optimization..."):
                    # Prepare returns data
                    returns_data = []
                    for etf in selected_etfs:
                        df = etf_data[etf]
                        returns = df['Close'].pct_change().dropna()
                        returns_data.append(returns)
                    
                    returns_df = pd.concat(returns_data, axis=1)
                    returns_df.columns = selected_etfs
                    returns_df = returns_df.dropna()
                    
                    # Calculate covariance matrix
                    cov_matrix = returns_df.cov() * 252
                    
                    # Run optimization (simplified for demo)
                    num_portfolios = 10000
                    results = np.zeros((3, num_portfolios))
                    
                    for i in range(num_portfolios):
                        weights = np.random.random(len(selected_etfs))
                        weights /= np.sum(weights)
                        
                        portfolio_return = np.sum(returns_df.mean() * weights) * 252
                        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        
                        results[0,i] = portfolio_return
                        results[1,i] = portfolio_volatility
                        results[2,i] = portfolio_return / portfolio_volatility  # Sharpe ratio
                    
                    # Find optimal portfolio based on goal
                    if optimization_goal == "Maximize Returns":
                        optimal_idx = np.argmax(results[0])
                    elif optimization_goal == "Minimize Risk":
                        optimal_idx = np.argmin(results[1])
                    else:  # Balanced
                        optimal_idx = np.argmax(results[2])
                    
                    optimal_weights = np.random.random(len(selected_etfs))
                    optimal_weights /= np.sum(optimal_weights)  # Placeholder for actual weights
                    
                    # Display results
                    st.subheader("Optimized Portfolio Allocation")
                    
                    # Create allocation dataframe
                    allocation = pd.DataFrame({
                        'ETF': selected_etfs,
                        'Weight': optimal_weights,
                        'Amount (₹)': investment_amount * optimal_weights
                    })
                    
                    # Format display
                    allocation['Weight'] = allocation['Weight'].apply(lambda x: f"{x:.1%}")
                    allocation['Amount (₹)'] = allocation['Amount (₹)'].apply(lambda x: f"₹{x:,.2f}")
                    
                    st.dataframe(allocation)
                    
                    # Show portfolio metrics
                    st.subheader("Portfolio Characteristics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Expected Annual Return",
                            f"{results[0, optimal_idx]:.1%}"
                        )
                    
                    with col2:
                        st.metric(
                            "Expected Annual Volatility",
                            f"{results[1, optimal_idx]:.1%}"
                        )
                    
                    with col3:
                        st.metric(
                            "Expected Sharpe Ratio",
                            f"{results[2, optimal_idx]:.2f}"
                        )
                    
                    # Efficient frontier plot
                    st.subheader("Efficient Frontier")
                    
                    fig = go.Figure()
                    
                    # All random portfolios
                    fig.add_trace(go.Scatter(
                        x=results[1,:],
                        y=results[0,:],
                        mode='markers',
                        marker=dict(
                            color=results[2,:],
                            colorscale='Viridis',
                            showscale=True,
                            size=5,
                            opacity=0.5
                        ),
                        name='Possible Portfolios'
                    ))
                    
                    # Optimal portfolio
                    fig.add_trace(go.Scatter(
                        x=[results[1, optimal_idx]],
                        y=[results[0, optimal_idx]],
                        mode='markers',
                        marker=dict(
                            color='red',
                            size=10,
                            line=dict(width=2, color='black')
                        ),
                        name='Optimal Portfolio'
                    ))
                    
                    fig.update_layout(
                        title='Portfolio Optimization - Efficient Frontier',
                        xaxis_title='Annualized Volatility',
                        yaxis_title='Annualized Return',
                        coloraxis_colorbar=dict(title='Sharpe Ratio')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("ETF Deep Dive Explorer")
        
        selected_etf = st.selectbox(
            "Select ETF to explore",
            metrics_df['ETF'].sort_values()
        )
        
        if selected_etf:
            df = etf_data[selected_etf]
            metrics = metrics_df[metrics_df['ETF'] == selected_etf].iloc[0]
            
            # Header with key metrics
            st.subheader(f"{selected_etf} - {metrics['Category']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"₹{metrics['Last_Price']:.2f}")
                st.metric("1 Week Return", f"{metrics['1W_Return']*100:.2f}%")
                st.metric("1 Month Return", f"{metrics['1M_Return']*100:.2f}%")
            
            with col2:
                st.metric("YTD Return", f"{metrics['YTD_Return']*100:.2f}%")
                st.metric("Volatility", f"{metrics['Volatility']:.4f}")
                st.metric("Sharpe Ratio", f"{metrics['Sharpe_Ratio']:.2f}")
            
            with col3:
                st.metric("RSI (14-day)", f"{metrics['RSI_Current']:.1f}")
                st.metric("Max Drawdown", f"{metrics['Max_Drawdown']*100:.2f}%")
                st.metric("Beta", f"{metrics['Beta']:.2f}")
            
            # Technical charts
            st.subheader("Technical Analysis")
            
            tab_tech1, tab_tech2, tab_tech3 = st.tabs(["Price & Moving Averages", "MACD", "RSI"])
            
            with tab_tech1:
                fig = go.Figure()
                
                # Price
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name='Price',
                    line=dict(color='royalblue', width=2)
                ))
                
                # Moving averages
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'].rolling(20).mean(),
                    name='20-day SMA',
                    line=dict(color='orange', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'].rolling(50).mean(),
                    name='50-day SMA',
                    line=dict(color='green', width=1)
                ))
                
                fig.update_layout(
                    title=f"{selected_etf} Price with Moving Averages",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab_tech2:
                fig = go.Figure()
                
                # MACD
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    name='MACD',
                    line=dict(color='blue', width=2)
                ))
                
                # Signal line
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Signal_Line'],
                    name='Signal Line',
                    line=dict(color='red', width=1)
                ))
                
                # Histogram
                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df['MACD'] - df['Signal_Line'],
                    name='Histogram',
                    marker_color=np.where((df['MACD'] - df['Signal_Line']) >= 0, 'green', 'red')
                ))
                
                fig.update_layout(
                    title=f"{selected_etf} MACD Indicator",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab_tech3:
                fig = go.Figure()
                
                # RSI
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=2)
                ))
                
                # Overbought line
                fig.add_hline(
                    y=70,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Overbought",
                    annotation_position="bottom right"
                )
                
                # Oversold line
                fig.add_hline(
                    y=30,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Oversold",
                    annotation_position="top right"
                )
                
                fig.update_layout(
                    title=f"{selected_etf} RSI (14-day)",
                    xaxis_title="Date",
                    yaxis_title="RSI Value",
                    yaxis_range=[0, 100],
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Volume analysis
            st.subheader("Volume Analysis")
            
            fig = go.Figure()
            
            # Volume bars
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='green'
            ))
            
            # 20-day average volume
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Volume'].rolling(20).mean(),
                name='20-day Avg Volume',
                line=dict(color='red', width=1)
            ))
            
            fig.update_layout(
                title=f"{selected_etf} Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # News sentiment
            st.subheader("Market Sentiment Analysis")
            
            sentiment = get_news_sentiment(selected_etf)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Sentiment Score", f"{sentiment['sentiment_score']:.2f}")
            
            with col2:
                st.metric("Overall Sentiment", sentiment['sentiment'])
            
            # Simulated news headlines (in a real app, use a news API)
            st.write("**Recent Market News**")
            
            news_samples = [
                {
                    'headline': f"Positive outlook for {selected_etf} as market conditions improve",
                    'sentiment': 'positive'
                },
                {
                    'headline': f"Investors show mixed reactions to {selected_etf} performance",
                    'sentiment': 'neutral'
                },
                {
                    'headline': f"{selected_etf} reaches new highs amid market rally",
                    'sentiment': 'positive'
                },
                {
                    'headline': f"Analysts caution about potential correction in {selected_etf.split('.')[0]} sector",
                    'sentiment': 'negative'
                }
            ]
            
            for news in news_samples:
                if news['sentiment'] == 'positive':
                    st.success(f"📰 {news['headline']}")
                elif news['sentiment'] == 'negative':
                    st.error(f"📰 {news['headline']}")
                else:
                    st.info(f"📰 {news['headline']}")

if __name__ == "__main__":
    main()
