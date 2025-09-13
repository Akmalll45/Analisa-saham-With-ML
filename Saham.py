import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Indonesia Stock Analysis Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem 1rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .ml-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    .warren-analysis {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🇮🇩 INDONESIA STOCK ANALYSIS PRO</h1>
    <p>Advanced Machine Learning & Real-time Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

# Daftar saham Indonesia
INDONESIA_STOCKS = {
    "🏦 Perbankan": {
        "BBCA.JK": "Bank Central Asia",
        "BBRI.JK": "Bank Rakyat Indonesia", 
        "BMRI.JK": "Bank Mandiri",
        "BBNI.JK": "Bank Negara Indonesia",
        "BRIS.JK": "Bank BRI Syariah",
        "BDMN.JK": "Bank Danamon"
    },
    "🏭 Industri & Manufaktur": {
        "ASII.JK": "Astra International",
        "UNTR.JK": "United Tractors",
        "INTP.JK": "Indocement Tunggal Prakarsa",
        "SMGR.JK": "Semen Indonesia",
        "TINS.JK": "Timah",
        "ANTM.JK": "Aneka Tambang"
    },
    "🛒 Konsumen": {
        "UNVR.JK": "Unilever Indonesia",
        "ICBP.JK": "Indofood CBP Sukses Makmur",
        "INDF.JK": "Indofood Sukses Makmur",
        "GGRM.JK": "Gudang Garam",
        "HMSP.JK": "HM Sampoerna",
        "KLBF.JK": "Kalbe Farma"
    },
    "📡 Telekomunikasi": {
        "TLKM.JK": "Telkom Indonesia",
        "EXCL.JK": "XL Axiata",
        "ISAT.JK": "Indosat Ooredoo Hutchison"
    },
    "🏢 Properti": {
        "BSDE.JK": "Bumi Serpong Damai",
        "LPKR.JK": "Lippo Karawaci",
        "SMRA.JK": "Summarecon Agung"
    },
    "⛽ Energi & Tambang": {
        "PGAS.JK": "Perusahaan Gas Negara",
        "PTBA.JK": "Bukit Asam",
        "ADRO.JK": "Adaro Energy"
    }
}

# Sidebar
st.sidebar.title("🎯 STOCK SELECTION")

selected_sector = st.sidebar.selectbox(
    "📊 Pilih Sektor:", 
    list(INDONESIA_STOCKS.keys())
)

stocks_in_sector = INDONESIA_STOCKS[selected_sector]
selected_stock_code = st.sidebar.selectbox(
    "📈 Pilih Saham:",
    list(stocks_in_sector.keys()),
    format_func=lambda x: f"{x} - {stocks_in_sector[x]}"
)

custom_stock = st.sidebar.text_input("✍️ Atau masukkan kode saham manual:")
if custom_stock:
    stock_symbol = custom_stock.upper()
    if not stock_symbol.endswith('.JK'):
        stock_symbol += '.JK'
else:
    stock_symbol = selected_stock_code

st.sidebar.title("⚙️ ANALYSIS SETTINGS")
period = st.sidebar.selectbox(
    "📅 Periode:", 
    ["1mo", "3mo", "6mo", "1y", "2y"],
    index=2
)

ml_model_type = st.sidebar.selectbox(
    "🤖 Model ML:",
    ["Random Forest", "Gradient Boosting", "Linear Regression"]
)

ml_prediction_days = st.sidebar.slider(
    "📊 Hari Prediksi:", 1, 30, 10
)

# DEFINISI SEMUA FUNGSI TERLEBIH DAHULU
@st.cache_data(ttl=300)
def get_stock_data(symbol, period):
    """Ambil data saham"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, auto_adjust=True)
        info = stock.info
        
        if data.empty:
            return None, None
            
        return data, info
    except Exception as e:
        st.error(f"Error mengambil data: {str(e)}")
        return None, None

def calculate_rsi(prices, window=14):
    """Hitung RSI dengan error handling yang lebih baik"""
    try:
        if len(prices) < window:
            window = max(2, len(prices) // 2)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        
        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    except Exception as e:
        return pd.Series([50] * len(prices), index=prices.index)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Hitung MACD dengan error handling"""
    try:
        if len(prices) < slow:
            fast = min(fast, len(prices) // 3)
            slow = min(slow, len(prices) // 2)
            signal = min(signal, len(prices) // 4)
        
        ema_fast = prices.ewm(span=fast, min_periods=1).mean()
        ema_slow = prices.ewm(span=slow, min_periods=1).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, min_periods=1).mean()
        
        return macd, macd_signal
    except Exception as e:
        zeros = pd.Series([0] * len(prices), index=prices.index)
        return zeros, zeros

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Hitung Bollinger Bands dengan error handling"""
    try:
        if len(prices) < window:
            window = max(5, len(prices) // 2)
        
        rolling_mean = prices.rolling(window=window, min_periods=1).mean()
        rolling_std = prices.rolling(window=window, min_periods=1).std()
        
        rolling_std = rolling_std.fillna(rolling_mean * 0.02)
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band, rolling_mean, lower_band
    except Exception as e:
        return prices, prices, prices

def calculate_technical_indicators(data):
    """Hitung semua indikator teknikal dengan error handling"""
    try:
        df = data.copy()
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Kolom {col} tidak ditemukan dalam data")
                return data
        
        if len(df) < 50:
            st.warning("Data tidak cukup untuk semua indikator, menggunakan indikator sederhana")
            min_window = min(10, len(df) // 2)
        else:
            min_window = 10
        
        # Moving averages
        if len(df) >= 10:
            df['SMA_10'] = df['Close'].rolling(window=min_window, min_periods=1).mean()
        if len(df) >= 20:
            df['SMA_20'] = df['Close'].rolling(window=min(20, len(df)), min_periods=1).mean()
        if len(df) >= 50:
            df['SMA_50'] = df['Close'].rolling(window=min(50, len(df)), min_periods=1).mean()
        
        # RSI
        try:
            df['RSI'] = calculate_rsi(df['Close'])
        except Exception as e:
            st.warning(f"Error calculating RSI: {e}")
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        try:
            df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'])
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        except Exception as e:
            st.warning(f"Error calculating MACD: {e}")
            ema12 = df['Close'].ewm(span=12, min_periods=1).mean()
            ema26 = df['Close'].ewm(span=26, min_periods=1).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        try:
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df['Close'])
        except Exception as e:
            st.warning(f"Error calculating Bollinger Bands: {e}")
            window = min(20, len(df))
            df['BB_middle'] = df['Close'].rolling(window=window, min_periods=1).mean()
            bb_std = df['Close'].rolling(window=window, min_periods=1).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volume indicators
        try:
            volume_window = min(20, len(df))
            df['Volume_SMA'] = df['Volume'].rolling(window=volume_window, min_periods=1).mean()
            df['Volume_ratio'] = df['Volume'] / (df['Volume_SMA'] + 1e-10)
        except Exception as e:
            st.warning(f"Error calculating volume indicators: {e}")
            df['Volume_SMA'] = df['Volume'].rolling(window=5, min_periods=1).mean()
            df['Volume_ratio'] = df['Volume'] / (df['Volume_SMA'] + 1e-10)
        
        # Price features
        try:
            df['Price_change'] = df['Close'].pct_change()
            df['Volatility'] = df['Price_change'].rolling(20, min_periods=1).std()
            df['High_Low_ratio'] = df['High'] / (df['Low'] + 1e-10)
        except Exception as e:
            st.warning(f"Error calculating price features: {e}")
            df['Price_change'] = df['Close'].diff() / df['Close'].shift(1)
            df['Volatility'] = df['Price_change'].rolling(10, min_periods=1).std()
            df['High_Low_ratio'] = df['High'] / (df['Low'] + 1e-10)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    except Exception as e:
        st.error(f"Error dalam calculate_technical_indicators: {str(e)}")
        st.info("Menggunakan data asli tanpa indikator teknikal")
        return data

def analyze_trend(data):
    """Analyze price trend"""
    try:
        recent_data = data['Close'].tail(20).values
        x = np.arange(len(recent_data))
        slope, intercept = np.polyfit(x, recent_data, 1)
        
        trend_pct = (slope * len(recent_data)) / recent_data[0] * 100
        
        if trend_pct > 2:
            main_trend = "🟢 UPTREND KUAT"
            trend_strength = "Kuat"
        elif trend_pct > 0.5:
            main_trend = "📈 UPTREND"  
            trend_strength = "Sedang"
        elif trend_pct > -0.5:
            main_trend = "➡️ SIDEWAYS"
            trend_strength = "Lemah"
        elif trend_pct > -2:
            main_trend = "📉 DOWNTREND"
            trend_strength = "Sedang"
        else:
            main_trend = "🔴 DOWNTREND KUAT"
            trend_strength = "Kuat"
        
        trend_duration = count_trend_duration(data['Close'])
        
        return {
            'main_trend': main_trend,
            'trend_strength': trend_strength,
            'trend_duration': trend_duration,
            'trend_pct': trend_pct
        }
    except:
        return {
            'main_trend': "Data tidak cukup",
            'trend_strength': "Unknown", 
            'trend_duration': 0,
            'trend_pct': 0
        }

def count_trend_duration(prices):
    """Count trend duration"""
    try:
        if len(prices) < 10:
            return 0
        
        sma = prices.rolling(10).mean()
        current_direction = 1 if sma.iloc[-1] > sma.iloc[-2] else -1
        
        count = 1
        for i in range(len(sma)-2, 0, -1):
            if len(sma) > i+1:
                direction = 1 if sma.iloc[i] > sma.iloc[i-1] else -1
                if direction == current_direction:
                    count += 1
                else:
                    break
        
        return min(count, len(prices))
    except:
        return 0

def find_support_resistance(data):
    """Find support and resistance levels"""
    try:
        if len(data) < 20:
            return None
        
        highs = data['High'].tail(50)
        lows = data['Low'].tail(50)
        current_price = data['Close'].iloc[-1]
        
        resistance_levels = highs[highs > current_price].sort_values(ascending=False)
        resistance = resistance_levels.iloc[0] if len(resistance_levels) > 0 else current_price * 1.05
        
        support_levels = lows[lows < current_price].sort_values(ascending=False)
        support = support_levels.iloc[0] if len(support_levels) > 0 else current_price * 0.95
        
        distance_to_resistance = ((resistance - current_price) / current_price) * 100
        distance_to_support = ((current_price - support) / current_price) * 100
        
        return {
            'resistance': resistance,
            'support': support,
            'distance_to_resistance': distance_to_resistance,
            'distance_to_support': distance_to_support
        }
    except:
        return None

def analyze_volume(data):
    """Analyze volume patterns"""
    try:
        if len(data) < 20:
            return {'trend': 'Data tidak cukup', 'vs_average': 'Unknown', 'signal': 'Neutral'}
        
        recent_volume = data['Volume'].tail(10).mean()
        older_volume = data['Volume'].tail(30).head(20).mean()
        
        if recent_volume > older_volume * 1.2:
            volume_trend = "📈 MENINGKAT"
        elif recent_volume < older_volume * 0.8:
            volume_trend = "📉 MENURUN"
        else:
            volume_trend = "➡️ STABIL"
        
        avg_volume = data['Volume'].tail(50).mean()
        current_volume = data['Volume'].iloc[-1]
        
        if current_volume > avg_volume * 1.5:
            vs_average = "🔥 TINGGI (>150% avg)"
        elif current_volume > avg_volume * 1.2:
            vs_average = "📊 ATAS RATA-RATA"
        elif current_volume < avg_volume * 0.5:
            vs_average = "📉 RENDAH (<50% avg)"
        else:
            vs_average = "➡️ NORMAL"
        
        price_change = data['Close'].pct_change().iloc[-1]
        if current_volume > avg_volume * 1.3:
            if price_change > 0:
                signal = "🟢 BULLISH (Volume tinggi + harga naik)"
            else:
                signal = "🔴 BEARISH (Volume tinggi + harga turun)"
        else:
            signal = "⚪ NETRAL (Volume normal)"
        
        return {
            'trend': volume_trend,
            'vs_average': vs_average,
            'signal': signal
        }
    except:
        return {'trend': 'Error', 'vs_average': 'Error', 'signal': 'Error'}

def analyze_momentum(data):
    """Analyze momentum indicators"""
    momentum = {}
    
    try:
        if 'RSI' in data.columns:
            rsi = data['RSI'].iloc[-1]
            if rsi > 70:
                momentum['RSI'] = f"🔴 OVERBOUGHT ({rsi:.1f})"
            elif rsi < 30:
                momentum['RSI'] = f"🟢 OVERSOLD ({rsi:.1f})" 
            else:
                momentum['RSI'] = f"⚪ NEUTRAL ({rsi:.1f})"
        
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            macd = data['MACD'].iloc[-1]
            macd_signal = data['MACD_signal'].iloc[-1]
            
            if macd > macd_signal:
                momentum['MACD'] = "🟢 BULLISH (MACD > Signal)"
            else:
                momentum['MACD'] = "🔴 BEARISH (MACD < Signal)"
        
        if len(data) >= 10:
            roc = ((data['Close'].iloc[-1] - data['Close'].iloc[-10]) / data['Close'].iloc[-10]) * 100
            if roc > 5:
                momentum['Price ROC'] = f"🟢 STRONG UP ({roc:.1f}%)"
            elif roc > 2:
                momentum['Price ROC'] = f"📈 UP ({roc:.1f}%)"
            elif roc > -2:
                momentum['Price ROC'] = f"➡️ FLAT ({roc:.1f}%)"
            elif roc > -5:
                momentum['Price ROC'] = f"📉 DOWN ({roc:.1f}%)"
            else:
                momentum['Price ROC'] = f"🔴 STRONG DOWN ({roc:.1f}%)"
        
        return momentum
    except:
        return {'Error': 'Cannot analyze momentum'}

def analyze_price_action(data):
    """Analyze price action patterns"""
    try:
        actions = []
        
        if len(data) < 10:
            return ['Data tidak cukup untuk analisis']
        
        recent_candles = data.tail(5)
        
        green_candles = 0
        red_candles = 0
        
        for _, candle in recent_candles.iterrows():
            if candle['Close'] > candle['Open']:
                green_candles += 1
                red_candles = 0
            else:
                red_candles += 1
                green_candles = 0
        
        if green_candles >= 3:
            actions.append(f"🟢 {green_candles} candle hijau berturut-turut - Momentum bullish")
        elif red_candles >= 3:
            actions.append(f"🔴 {red_candles} candle merah berturut-turut - Momentum bearish")
        
        last_close = data['Close'].iloc[-2]
        current_open = data['Open'].iloc[-1]
        gap_pct = ((current_open - last_close) / last_close) * 100
        
        if abs(gap_pct) > 2:
            if gap_pct > 0:
                actions.append(f"📈 Gap up {gap_pct:.1f}% - Bullish signal")
            else:
                actions.append(f"📉 Gap down {gap_pct:.1f}% - Bearish signal")
        
        last_candle = data.iloc[-1]
        body_size = abs(last_candle['Close'] - last_candle['Open']) / last_candle['Open'] * 100
        
        if body_size < 0.5:
            actions.append("⚡ Doji detected - Indecision/possible reversal")
        
        return actions if actions else ['Tidak ada pola khusus terdeteksi']
    
    except:
        return ['Error dalam analisis price action']

def analyze_chart_patterns(data):
    """Detect chart patterns"""
    try:
        patterns = []
        
        if len(data) < 20:
            return {'patterns': ['Data tidak cukup untuk deteksi pattern']}
        
        if 'SMA_10' in data.columns and 'SMA_20' in data.columns:
            sma10_current = data['SMA_10'].iloc[-1]
            sma20_current = data['SMA_20'].iloc[-1]
            sma10_prev = data['SMA_10'].iloc[-2]
            sma20_prev = data['SMA_20'].iloc[-2]
            
            if sma10_current > sma20_current and sma10_prev <= sma20_prev:
                patterns.append("🟢 Golden Cross - SMA 10 cross above SMA 20 (Bullish)")
            elif sma10_current < sma20_current and sma10_prev >= sma20_prev:
                patterns.append("🔴 Death Cross - SMA 10 cross below SMA 20 (Bearish)")
        
        if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
            bb_width_current = data['BB_upper'].iloc[-1] - data['BB_lower'].iloc[-1]
            bb_width_avg = (data['BB_upper'] - data['BB_lower']).tail(20).mean()
            
            if bb_width_current < bb_width_avg * 0.7:
                patterns.append("🔗 Bollinger Band Squeeze - Volatility rendah, breakout coming")
            elif bb_width_current > bb_width_avg * 1.3:
                patterns.append("💥 Bollinger Band Expansion - High volatility period")
        
        support_resistance = find_support_resistance(data)
        if support_resistance:
            current_price = data['Close'].iloc[-1]
            
            if support_resistance['distance_to_resistance'] < 3:
                patterns.append(f"⚠️ Mendekati Resistance (Rp {support_resistance['resistance']:,.0f})")
            
            if support_resistance['distance_to_support'] < 3:
                patterns.append(f"⚠️ Mendekati Support (Rp {support_resistance['support']:,.0f})")
        
        return {'patterns': patterns if patterns else ['Tidak ada pola signifikan terdeteksi']}
    
    except:
        return {'patterns': ['Error dalam deteksi pattern']}

def determine_overall_signal(data, chart_analysis, trend_analysis, volume_analysis, momentum_analysis):
    """Determine overall signal"""
    try:
        bullish_signals = 0
        bearish_signals = 0
        reasons = []
        
        if 'UPTREND' in trend_analysis['main_trend']:
            bullish_signals += 2
            reasons.append("Trend utama bullish")
        elif 'DOWNTREND' in trend_analysis['main_trend']:
            bearish_signals += 2
            reasons.append("Trend utama bearish")
        
        if 'BULLISH' in volume_analysis['signal']:
            bullish_signals += 1
            reasons.append("Volume mendukung kenaikan")
        elif 'BEARISH' in volume_analysis['signal']:
            bearish_signals += 1
            reasons.append("Volume mendukung penurunan")
        
        bullish_momentum = 0
        bearish_momentum = 0
        
        for indicator, value in momentum_analysis.items():
            if 'BULLISH' in value or 'OVERSOLD' in value or 'UP' in value:
                bullish_momentum += 1
            elif 'BEARISH' in value or 'OVERBOUGHT' in value or 'DOWN' in value:
                bearish_momentum += 1
        
        if bullish_momentum > bearish_momentum:
            bullish_signals += 1
            reasons.append("Momentum indikator bullish")
        elif bearish_momentum > bullish_momentum:
            bearish_signals += 1
            reasons.append("Momentum indikator bearish")
        
        for pattern in chart_analysis['patterns']:
            if any(word in pattern for word in ['Golden Cross', 'Bullish', 'BULLISH']):
                bullish_signals += 1
                reasons.append("Pattern bullish terdeteksi")
            elif any(word in pattern for word in ['Death Cross', 'Bearish', 'BEARISH']):
                bearish_signals += 1
                reasons.append("Pattern bearish terdeteksi")
        
        total_signals = bullish_signals + bearish_signals
        
        if total_signals == 0:
            return {'direction': 'NEUTRAL', 'confidence': 50, 'reasons': ['Sinyal campuran/tidak jelas']}
        
        if bullish_signals > bearish_signals:
            confidence = min(90, 50 + (bullish_signals - bearish_signals) * 10)
            return {'direction': 'BULLISH', 'confidence': confidence, 'reasons': reasons}
        elif bearish_signals > bullish_signals:
            confidence = min(90, 50 + (bearish_signals - bullish_signals) * 10)
            return {'direction': 'BEARISH', 'confidence': confidence, 'reasons': reasons}
        else:
            return {'direction': 'NEUTRAL', 'confidence': 50, 'reasons': ['Sinyal bullish dan bearish seimbang']}
    
    except:
        return {'direction': 'ERROR', 'confidence': 0, 'reasons': ['Error dalam analisis']}

def generate_trading_recommendations(data, overall_signal, support_resistance):
    """Generate trading recommendations"""
    try:
        current_price = data['Close'].iloc[-1]
        
        entry_points = []
        risk_management = []
        price_targets = []
        
        if overall_signal['direction'] == 'BULLISH':
            entry_points.append(f"✅ Buy pada harga saat ini: Rp {current_price:,.0f}")
            
            if support_resistance:
                entry_points.append(f"✅ Buy on dip near support: Rp {support_resistance['support']:,.0f}")
                
                target1 = current_price * 1.05
                target2 = support_resistance['resistance']
                price_targets.append(f"🎯 Target 1: Rp {target1:,.0f} (+5%)")
                price_targets.append(f"🎯 Target 2: Rp {target2:,.0f} (Resistance)")
                
                stop_loss = support_resistance['support'] * 0.98
                risk_management.append(f"🛡️ Stop Loss: Rp {stop_loss:,.0f}")
            
            risk_management.append("📊 Position size: Maksimal 5% dari portfolio")
            risk_management.append("⏰ Hold period: 1-4 minggu")
        
        elif overall_signal['direction'] == 'BEARISH':
            entry_points.append("⚠️ Avoid buying, consider selling")
            entry_points.append("📉 Short opportunity jika ada")
            
            if support_resistance:
                entry_points.append(f"🔄 Re-evaluate jika bounce dari support: Rp {support_resistance['support']:,.0f}")
                
                target1 = current_price * 0.95
                target2 = support_resistance['support']
                price_targets.append(f"📉 Downside target 1: Rp {target1:,.0f} (-5%)")
                price_targets.append(f"📉 Downside target 2: Rp {target2:,.0f} (Support)")
            
            risk_management.append("🚫 Hindari menambah posisi")
            risk_management.append("💰 Consider profit taking jika ada profit")
            risk_management.append("⏰ Review dalam 1-2 minggu")
        
        else:
            entry_points.append("⏸️ Wait and see approach")
            entry_points.append("👀 Monitor untuk breakout")
            
            if support_resistance:
                entry_points.append(f"📈 Buy jika break resistance: Rp {support_resistance['resistance']:,.0f}")
                entry_points.append(f"📉 Sell jika break support: Rp {support_resistance['support']:,.0f}")
            
            risk_management.append("💼 Maintain current position")
            risk_management.append("🔍 Watch for clear signals")
            price_targets.append("🎯 Wait for direction confirmation")
        
        return {
            'entry_points': entry_points if entry_points else ['Tidak ada entry point jelas'],
            'risk_management': risk_management if risk_management else ['Gunakan risk management standar'],
            'price_targets': price_targets if price_targets else ['Target akan ditentukan setelah ada konfirmasi']
        }
    
    except:
        return {
            'entry_points': ['Error generating recommendations'],
            'risk_management': ['Use standard risk management'],
            'price_targets': ['Error calculating targets']
        }

def prepare_ml_features(data):
    """Siapkan features untuk ML"""
    try:
        if len(data) < 60:
            raise ValueError("Data tidak cukup untuk ML")
        
        features = pd.DataFrame(index=data.index)
        
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close']).diff()
        features['volatility'] = features['returns'].rolling(10).std()
        
        indicators = ['SMA_10', 'SMA_20', 'RSI', 'MACD', 'MACD_signal', 'Volume_ratio']
        for indicator in indicators:
            if indicator in data.columns:
                features[indicator] = data[indicator]
                features[f'{indicator}_pct'] = data[indicator].pct_change()
        
        if 'SMA_20' in data.columns:
            features['price_vs_sma20'] = (data['Close'] / data['SMA_20']) - 1
        
        for lag in [1, 2, 3]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            if 'RSI' in data.columns:
                features[f'rsi_lag_{lag}'] = data['RSI'].shift(lag)
        
        for window in [5, 10]:
            features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
            features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
        
        features_clean = features.dropna()
        
        if len(features_clean) < 30:
            raise ValueError("Data bersih tidak cukup")
        
        return features_clean
        
    except Exception as e:
        st.error(f"Error prepare features: {str(e)}")
        return None

def train_ml_model(data, features, model_type):
    """Train ML model"""
    try:
        target = data['Close'].shift(-1)
        
        min_len = min(len(features), len(target))
        X = features.iloc[:min_len-1]
        y = target.iloc[:min_len-1]
        
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 20:
            raise ValueError("Data training tidak cukup")
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        else:
            model = LinearRegression()
        
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                      cv=min(5, len(X_train)//5), 
                                      scoring='neg_mean_absolute_error')
            cv_score = -cv_scores.mean()
        except:
            cv_score = mae
        
        return {
            'model': model,
            'scaler': scaler,
            'features': X,
            'metrics': {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'cv_score': cv_score
            },
            'test_data': {
                'y_test': y_test,
                'y_pred': y_pred
            }
        }
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def predict_future(model_info, data, days):
    """Prediksi harga masa depan"""
    try:
        model = model_info['model']
        scaler = model_info['scaler']
        features = model_info['features']
        
        last_row = features.iloc[-1:].copy()
        predictions = []
        current_price = data['Close'].iloc[-1]
        
        for day in range(days):
            scaled = scaler.transform(last_row)
            pred_price = model.predict(scaled)[0]
            predictions.append(pred_price)
            
            if 'returns' in last_row.columns:
                if day == 0:
                    new_return = (pred_price - current_price) / current_price
                else:
                    new_return = (pred_price - predictions[-2]) / predictions[-2]
                last_row.iloc[0, last_row.columns.get_loc('returns')] = new_return
        
        std_error = model_info['metrics']['rmse']
        confidence_intervals = []
        
        for pred in predictions:
            margin = 1.96 * std_error
            confidence_intervals.append((pred - margin, pred + margin))
        
        return predictions, confidence_intervals
        
    except Exception as e:
        st.error(f"Error prediksi: {str(e)}")
        return None, None

def warren_buffett_analysis(info, data):
    """Analisis Warren Buffett"""
    try:
        pe = info.get('trailingPE', 0)
        pb = info.get('priceToBook', 0)
        debt_equity = info.get('debtToEquity', 0)
        roe = info.get('returnOnEquity', 0)
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        
        score = 0
        factors = []
        
        if pe > 0:
            if pe < 15:
                score += 25
                factors.append(f"✅ PE Ratio bagus: {pe:.2f}")
            elif pe < 25:
                score += 15
                factors.append(f"⚠️ PE Ratio sedang: {pe:.2f}")
            else:
                factors.append(f"❌ PE Ratio tinggi: {pe:.2f}")
        
        if pb > 0:
            if pb < 2:
                score += 25
                factors.append(f"✅ PB Ratio bagus: {pb:.2f}")
            elif pb < 3:
                score += 15
                factors.append(f"⚠️ PB Ratio sedang: {pb:.2f}")
            else:
                factors.append(f"❌ PB Ratio tinggi: {pb:.2f}")
        
        if debt_equity >= 0:
            if debt_equity < 0.5:
                score += 20
                factors.append(f"✅ D/E Ratio rendah: {debt_equity:.2f}")
            elif debt_equity < 1.0:
                score += 10
                factors.append(f"⚠️ D/E Ratio sedang: {debt_equity:.2f}")
            else:
                factors.append(f"❌ D/E Ratio tinggi: {debt_equity:.2f}")
        
        if roe > 0:
            roe_pct = roe * 100
            if roe_pct > 15:
                score += 15
                factors.append(f"✅ ROE bagus: {roe_pct:.1f}%")
            elif roe_pct > 10:
                score += 10
                factors.append(f"⚠️ ROE sedang: {roe_pct:.1f}%")
            else:
                factors.append(f"❌ ROE rendah: {roe_pct:.1f}%")
        
        if dividend_yield > 3:
            score += 10
            factors.append(f"✅ Dividend tinggi: {dividend_yield:.1f}%")
        elif dividend_yield > 1:
            score += 5
            factors.append(f"⚠️ Dividend rendah: {dividend_yield:.1f}%")
        else:
            factors.append("ℹ️ Tidak ada dividend")
        
        if volatility < 25:
            score += 5
            factors.append(f"✅ Volatility rendah: {volatility:.1f}%")
        else:
            factors.append(f"❌ Volatility tinggi: {volatility:.1f}%")
        
        if score >= 80:
            recommendation = "🟢 STRONG BUY"
        elif score >= 60:
            recommendation = "🟡 BUY" 
        elif score >= 40:
            recommendation = "🟠 HOLD"
        else:
            recommendation = "🔴 SELL"
        
        return {
            'score': score,
            'recommendation': recommendation,
            'factors': factors,
            'metrics': {
                'pe': pe,
                'pb': pb,
                'debt_equity': debt_equity,
                'roe': roe * 100 if roe else 0,
                'dividend_yield': dividend_yield,
                'volatility': volatility
            }
        }
        
    except Exception as e:
        return {'error': str(e)}

def create_charts(data, predictions=None):
    """Buat chart"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price & Indicators', 'RSI', 'Volume'),
        row_heights=[0.6, 0.2, 0.2],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'], 
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ), row=1, col=1)
    
    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_20'],
            name='SMA 20', line=dict(color='orange')
        ), row=1, col=1)
    
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_50'],
            name='SMA 50', line=dict(color='blue')
        ), row=1, col=1)
    
    if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_upper'],
            line=dict(color='gray', dash='dash'),
            name='BB Upper', showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_lower'],
            line=dict(color='gray', dash='dash'),
            fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
            name='BB Lower', showlegend=False
        ), row=1, col=1)
    
    if predictions:
        future_dates = pd.date_range(start=data.index[-1], periods=len(predictions)+1)[1:]
        fig.add_trace(go.Scatter(
            x=future_dates, y=predictions,
            name='ML Prediction',
            line=dict(color='gold', width=3, dash='dot'),
            mode='lines+markers'
        ), row=1, col=1)
    
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            name='RSI', line=dict(color='purple')
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=data.index, y=data['Volume'],
        name='Volume', marker_color='lightblue', opacity=0.7
    ), row=3, col=1)
    
    fig.update_layout(
        title=f"Technical Analysis - {stock_symbol}",
        height=800,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    return fig

# MAIN ANALYSIS LOGIC
if st.button("🚀 ANALYZE STOCK", type="primary"):
    with st.spinner(f"Menganalisis {stock_symbol}..."):
        progress = st.progress(0)
        status = st.empty()
        
        # Get data
        status.text("📡 Mengambil data...")
        progress.progress(20)
        data, info = get_stock_data(stock_symbol, period)
        
        if data is not None and not data.empty:
            # Calculate indicators
            status.text("📊 Menghitung indikator...")
            progress.progress(40)
            data = calculate_technical_indicators(data)
            
            # Basic metrics
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
            
            # Display metrics
            st.markdown(f"## 📈 {stocks_in_sector.get(stock_symbol, stock_symbol)}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💰 Harga", f"Rp {current_price:,.0f}", 
                         f"{price_change:+,.0f} ({price_change_pct:+.2f}%)")
            
            with col2:
                st.metric("📊 Volume", f"{data['Volume'].iloc[-1]:,.0f}")
            
            with col3:
                if 'RSI' in data.columns:
                    rsi = data['RSI'].iloc[-1]
                    st.metric("🎯 RSI", f"{rsi:.1f}")
                    
            with col4:
                market_cap = info.get('marketCap', 0)
                if market_cap > 1e12:
                    cap_str = f"Rp {market_cap/1e12:.1f}T"
                elif market_cap > 1e9:
                    cap_str = f"Rp {market_cap/1e9:.1f}B"
                else:
                    cap_str = f"Rp {market_cap/1e6:.1f}M"
                st.metric("🏢 Market Cap", cap_str)
            
            # Advanced Chart Analysis
            st.markdown("---")
            st.markdown("## 📊 ANALISIS CHART MENDALAM")
            
            chart_analysis = analyze_chart_patterns(data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Analisis Trend & Pattern")
                
                trend_analysis = analyze_trend(data)
                st.markdown(f"**Trend Utama:** {trend_analysis['main_trend']}")
                st.markdown(f"**Kekuatan Trend:** {trend_analysis['trend_strength']}")
                st.markdown(f"**Durasi Trend:** {trend_analysis['trend_duration']} hari")
                
                support_resistance = find_support_resistance(data)
                if support_resistance:
                    st.markdown("**Support & Resistance:**")
                    st.markdown(f"• Resistance: Rp {support_resistance['resistance']:,.0f}")
                    st.markdown(f"• Support: Rp {support_resistance['support']:,.0f}")
                    st.markdown(f"• Distance to Resistance: {support_resistance['distance_to_resistance']:.1f}%")
                    st.markdown(f"• Distance to Support: {support_resistance['distance_to_support']:.1f}%")
                
                if chart_analysis['patterns']:
                    st.markdown("**Pola Chart Terdeteksi:**")
                    for pattern in chart_analysis['patterns']:
                        st.markdown(f"• {pattern}")
            
            with col2:
                st.markdown("### 📊 Analisis Volume & Momentum")
                
                volume_analysis = analyze_volume(data)
                st.markdown(f"**Volume Trend:** {volume_analysis['trend']}")
                st.markdown(f"**Volume vs Average:** {volume_analysis['vs_average']}")
                st.markdown(f"**Volume Signal:** {volume_analysis['signal']}")
                
                momentum_analysis = analyze_momentum(data)
                st.markdown("**Momentum Indicators:**")
                for indicator, value in momentum_analysis.items():
                    st.markdown(f"• {indicator}: {value}")
                
                price_action = analyze_price_action(data)
                st.markdown("**Price Action:**")
                for action in price_action:
                    st.markdown(f"• {action}")
            
            # Overall Signal
            st.markdown("---")
            st.markdown("## 💡 KESIMPULAN ANALISIS CHART")
            
            overall_signal = determine_overall_signal(data, chart_analysis, trend_analysis, volume_analysis, momentum_analysis)
            
            if overall_signal['direction'] == 'BULLISH':
                st.success(f"🟢 **SINYAL KESELURUHAN: BULLISH** (Confidence: {overall_signal['confidence']}%)")
            elif overall_signal['direction'] == 'BEARISH':
                st.error(f"🔴 **SINYAL KESELURUHAN: BEARISH** (Confidence: {overall_signal['confidence']}%)")
            else:
                st.info(f"🟡 **SINYAL KESELURUHAN: NEUTRAL** (Confidence: {overall_signal['confidence']}%)")
            
            st.markdown("**Reasoning:**")
            for reason in overall_signal['reasons']:
                st.markdown(f"• {reason}")
            
            # Trading Recommendations
            st.markdown("### 🎯 REKOMENDASI TRADING")
            recommendations = generate_trading_recommendations(data, overall_signal, support_resistance)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📈 Entry Points:**")
                for entry in recommendations['entry_points']:
                    st.markdown(f"• {entry}")
            
            with col2:
                st.markdown("**🛡️ Risk Management:**")
                for risk in recommendations['risk_management']:
                    st.markdown(f"• {risk}")
            
            with col3:
                st.markdown("**🎯 Price Targets:**")
                for target in recommendations['price_targets']:
                    st.markdown(f"• {target}")
            
            # Machine Learning Analysis
            status.text("🤖 Training ML...")
            progress.progress(60)
            
            st.markdown('<div class="ml-section">', unsafe_allow_html=True)
            st.markdown("## 🤖 MACHINE LEARNING ANALYSIS")
            
            features = prepare_ml_features(data)
            
            if features is not None:
                st.info(f"📊 Dataset: {len(features)} samples, {len(features.columns)} features")
                
                ml_results = train_ml_model(data, features, ml_model_type)
                
                if ml_results:
                    metrics = ml_results['metrics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 R² Score", f"{metrics['r2']:.4f}")
                    with col2:
                        st.metric("📏 MAE", f"Rp {metrics['mae']:,.0f}")
                    with col3:
                        st.metric("📐 RMSE", f"Rp {metrics['rmse']:,.0f}")
                    with col4:
                        st.metric("✅ CV Score", f"Rp {metrics['cv_score']:,.0f}")
                    
                    r2 = metrics['r2']
                    if r2 > 0.7:
                        st.success("🟢 Model Performance: EXCELLENT")
                    elif r2 > 0.5:
                        st.info("🟡 Model Performance: GOOD")
                    elif r2 > 0.3:
                        st.warning("🟠 Model Performance: FAIR")
                    else:
                        st.error("🔴 Model Performance: POOR")
                    
                    status.text("🔮 Membuat prediksi...")
                    progress.progress(80)
                    
                    predictions, confidence_intervals = predict_future(ml_results, data, ml_prediction_days)
                    
                    if predictions:
                        st.markdown("### 📅 Prediksi Harga")
                        
                        future_dates = pd.date_range(start=data.index[-1], periods=ml_prediction_days+1)[1:]
                        pred_df = pd.DataFrame({
                            'Tanggal': future_dates.strftime('%Y-%m-%d'),
                            'Prediksi': [f"Rp {p:,.0f}" for p in predictions],
                            'Perubahan': [f"{((p-current_price)/current_price)*100:+.2f}%" for p in predictions]
                        })
                        
                        if confidence_intervals:
                            pred_df['CI Lower'] = [f"Rp {ci[0]:,.0f}" for ci in confidence_intervals]
                            pred_df['CI Upper'] = [f"Rp {ci[1]:,.0f}" for ci in confidence_intervals]
                        
                        st.dataframe(pred_df, use_container_width=True)
                        
                        avg_pred = np.mean(predictions)
                        change = ((avg_pred - current_price) / current_price) * 100
                        
                        if change > 5:
                            st.success(f"🚀 Prediksi kenaikan {change:.1f}%")
                        elif change > 0:
                            st.info(f"📈 Prediksi kenaikan {change:.1f}%")
                        elif change > -5:
                            st.warning(f"📉 Prediksi penurunan {change:.1f}%")
                        else:
                            st.error(f"⚠️ Prediksi penurunan besar {change:.1f}%")
                        
                    else:
                        predictions = None
                else:
                    predictions = None
            else:
                st.warning("❌ Data tidak cukup untuk ML")
                predictions = None
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Warren Buffett Analysis  
            status.text(" Analisis ...")
            progress.progress(90)
            
            warren = warren_buffett_analysis(info, data)
            
            st.markdown('<div class="warren-analysis">', unsafe_allow_html=True)
            st.markdown("## ANALYSIS")
            
            if 'error' not in warren:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("📊 Score", f"{warren['score']}/100")
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = warren['score'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Score"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "gold"},
                            'steps': [
                                {'range': [0, 40], 'color': "red"},
                                {'range': [40, 60], 'color': "orange"}, 
                                {'range': [60, 80], 'color': "lightgreen"},
                                {'range': [80, 100], 'color': "green"}
                            ]
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎯 Rekomendasi")
                    st.markdown(f"**{warren['recommendation']}**")
                    
                    st.markdown("### 📋 Faktor:")
                    for factor in warren['factors']:
                        st.markdown(f"• {factor}")
                
                st.markdown("### 📊 Detail Fundamental")
                metrics = warren['metrics']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Valuasi**")
                    if metrics['pe'] > 0:
                        st.metric("P/E Ratio", f"{metrics['pe']:.2f}")
                    if metrics['pb'] > 0:
                        st.metric("P/B Ratio", f"{metrics['pb']:.2f}")
                
                with col2:
                    st.markdown("**Profitabilitas**") 
                    if metrics['roe'] > 0:
                        st.metric("ROE", f"{metrics['roe']:.1f}%")
                    st.metric("Dividend", f"{metrics['dividend_yield']:.1f}%")
                
                with col3:
                    st.markdown("**Risiko**")
                    st.metric("Volatility", f"{metrics['volatility']:.1f}%")
                    if metrics['debt_equity'] > 0:
                        st.metric("D/E Ratio", f"{metrics['debt_equity']:.2f}")
            
            else:
                st.error("❌ Error analisis fundamental")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Charts
            status.text("📊 Membuat chart...")
            progress.progress(95)
            
            try:
                chart = create_charts(data, predictions)
                st.plotly_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error chart: {str(e)}")
            
            progress.progress(100)
            status.text("✅ Analisis selesai!")
            time.sleep(1)
            progress.empty()
            status.empty()
        
        else:
            st.error(f"❌ Tidak dapat mengambil data untuk {stock_symbol}")

# Sidebar tips
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 TIPS")
st.sidebar.markdown("""
✅ Gunakan periode minimal 6 bulan
✅ Kombinasikan analisis teknikal & fundamental  
✅ Perhatikan volume konfirmasi
✅ Set stop loss dan take profit
❌ Jangan FOMO atau panic selling
""")

st.sidebar.markdown("### ⚠️ DISCLAIMER")
st.sidebar.markdown("""
Aplikasi ini untuk edukasi.
Bukan saran investasi.
Lakukan riset mandiri.
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p>🇮🇩 <strong>Indonesia Stock Analysis Pro</strong></p>
    <p>⚠️ Educational purpose only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
