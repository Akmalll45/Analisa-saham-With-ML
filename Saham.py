import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import warnings
import time
from datetime import datetime, timedelta
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import ta
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Enhanced page configuration
st.set_page_config(
    page_title="Warren Buffett Stock Analyzer Pro - Indonesia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern dark theme and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: brightness(1); }
        to { filter: brightness(1.1); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    .strong-buy {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        animation: pulse-green 2s infinite;
    }
    
    .buy {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
    }
    
    .hold {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border: none;
    }
    
    .sell {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border: none;
    }
    
    @keyframes pulse-green {
        0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        50% { transform: scale(1.03); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
    }
    
    .financial-health-excellent {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 6px solid #10b981;
    }
    
    .financial-health-good {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 6px solid #3b82f6;
    }
    
    .financial-health-poor {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 6px solid #ef4444;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
    }
    
    .sidebar .stSelectbox > div > div {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 10px;
    }
    
    .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Expanded Indonesian stocks with more comprehensive data
INDONESIAN_STOCKS = {
    'Perbankan': {
        'BBCA.JK': {'name': 'Bank Central Asia Tbk', 'tier': 'Tier 1', 'market_cap': 'Large'},
        'BBRI.JK': {'name': 'Bank Rakyat Indonesia Tbk', 'tier': 'Tier 1', 'market_cap': 'Large'}, 
        'BMRI.JK': {'name': 'Bank Mandiri Tbk', 'tier': 'Tier 1', 'market_cap': 'Large'},
        'BBNI.JK': {'name': 'Bank Negara Indonesia Tbk', 'tier': 'Tier 1', 'market_cap': 'Large'},
        'BDMN.JK': {'name': 'Bank Danamon Indonesia Tbk', 'tier': 'Tier 2', 'market_cap': 'Medium'},
        'BTPN.JK': {'name': 'Bank BTPN Tbk', 'tier': 'Tier 2', 'market_cap': 'Medium'},
        'MEGA.JK': {'name': 'Bank Mega Tbk', 'tier': 'Tier 2', 'market_cap': 'Medium'},
        'BNGA.JK': {'name': 'Bank CIMB Niaga Tbk', 'tier': 'Tier 2', 'market_cap': 'Medium'},
        'BNLI.JK': {'name': 'Bank Permata Tbk', 'tier': 'Tier 2', 'market_cap': 'Medium'},
        'BRIS.JK': {'name': 'Bank BRI Syariah Tbk', 'tier': 'Tier 2', 'market_cap': 'Small'}
    },
    'Konsumer': {
        'UNVR.JK': {'name': 'Unilever Indonesia Tbk', 'tier': 'Blue Chip', 'market_cap': 'Large'},
        'INDF.JK': {'name': 'Indofood Sukses Makmur Tbk', 'tier': 'Blue Chip', 'market_cap': 'Large'},
        'ICBP.JK': {'name': 'Indofood CBP Sukses Makmur Tbk', 'tier': 'Growth', 'market_cap': 'Large'},
        'KLBF.JK': {'name': 'Kalbe Farma Tbk', 'tier': 'Blue Chip', 'market_cap': 'Large'},
        'KAEF.JK': {'name': 'Kimia Farma Tbk', 'tier': 'BUMN', 'market_cap': 'Medium'},
        'MYOR.JK': {'name': 'Mayora Indah Tbk', 'tier': 'Growth', 'market_cap': 'Medium'},
        'GGRM.JK': {'name': 'Gudang Garam Tbk', 'tier': 'Dividend', 'market_cap': 'Large'},
        'HMSP.JK': {'name': 'HM Sampoerna Tbk', 'tier': 'Dividend', 'market_cap': 'Large'},
        'SIDO.JK': {'name': 'Industri Jamu Dan Farmasi Sido Muncul Tbk', 'tier': 'Growth', 'market_cap': 'Small'},
        'TCID.JK': {'name': 'Mandom Indonesia Tbk', 'tier': 'Growth', 'market_cap': 'Small'}
    },
    'Telekomunikasi': {
        'TLKM.JK': {'name': 'Telkom Indonesia Tbk', 'tier': 'BUMN', 'market_cap': 'Large'},
        'EXCL.JK': {'name': 'XL Axiata Tbk', 'tier': 'Growth', 'market_cap': 'Medium'},
        'ISAT.JK': {'name': 'Indosat Tbk', 'tier': 'Growth', 'market_cap': 'Medium'},
        'FREN.JK': {'name': 'Smartfren Telecom Tbk', 'tier': 'Speculative', 'market_cap': 'Small'},
        'BTEL.JK': {'name': 'Bakrie Telecom Tbk', 'tier': 'Speculative', 'market_cap': 'Small'}
    },
    'Pertambangan': {
        'ANTM.JK': {'name': 'Aneka Tambang Tbk', 'tier': 'BUMN', 'market_cap': 'Medium'},
        'INCO.JK': {'name': 'Vale Indonesia Tbk', 'tier': 'Commodity', 'market_cap': 'Large'},
        'TINS.JK': {'name': 'Timah Tbk', 'tier': 'BUMN', 'market_cap': 'Small'},
        'PTBA.JK': {'name': 'Bukit Asam Tbk', 'tier': 'BUMN', 'market_cap': 'Medium'},
        'ADRO.JK': {'name': 'Adaro Energy Tbk', 'tier': 'Commodity', 'market_cap': 'Large'},
        'ITMG.JK': {'name': 'Indo Tambangraya Megah Tbk', 'tier': 'Commodity', 'market_cap': 'Medium'},
        'HRUM.JK': {'name': 'Harum Energy Tbk', 'tier': 'Commodity', 'market_cap': 'Small'},
        'GEMS.JK': {'name': 'Golden Energy Mines Tbk', 'tier': 'Commodity', 'market_cap': 'Small'}
    },
    'Properti & Real Estate': {
        'BSDE.JK': {'name': 'Bumi Serpong Damai Tbk', 'tier': 'Blue Chip', 'market_cap': 'Large'},
        'LPKR.JK': {'name': 'Lippo Karawaci Tbk', 'tier': 'Recovery', 'market_cap': 'Medium'},
        'ASRI.JK': {'name': 'Alam Sutera Realty Tbk', 'tier': 'Growth', 'market_cap': 'Medium'},
        'PWON.JK': {'name': 'Pakuwon Jati Tbk', 'tier': 'Growth', 'market_cap': 'Medium'},
        'SMRA.JK': {'name': 'Summarecon Agung Tbk', 'tier': 'Blue Chip', 'market_cap': 'Medium'},
        'CTRA.JK': {'name': 'Ciputra Development Tbk', 'tier': 'Blue Chip', 'market_cap': 'Medium'},
        'DILD.JK': {'name': 'Intiland Development Tbk', 'tier': 'Recovery', 'market_cap': 'Small'},
        'KIJA.JK': {'name': 'Kawasan Industri Jababeka Tbk', 'tier': 'Growth', 'market_cap': 'Small'}
    },
    'Energi': {
        'PGAS.JK': {'name': 'Perusahaan Gas Negara Tbk', 'tier': 'BUMN', 'market_cap': 'Medium'},
        'AKRA.JK': {'name': 'AKR Corporindo Tbk', 'tier': 'Blue Chip', 'market_cap': 'Medium'},
        'MEDC.JK': {'name': 'Medco Energi International Tbk', 'tier': 'Commodity', 'market_cap': 'Medium'},
        'ELSA.JK': {'name': 'Elnusa Tbk', 'tier': 'BUMN', 'market_cap': 'Small'},
        'ENRG.JK': {'name': 'Energi Mega Persada Tbk', 'tier': 'Commodity', 'market_cap': 'Small'}
    },
    'Infrastruktur & Konstruksi': {
        'JSMR.JK': {'name': 'Jasa Marga Tbk', 'tier': 'BUMN', 'market_cap': 'Large'},
        'WIKA.JK': {'name': 'Wijaya Karya Tbk', 'tier': 'BUMN', 'market_cap': 'Medium'},
        'WSKT.JK': {'name': 'Waskita Karya Tbk', 'tier': 'BUMN', 'market_cap': 'Medium'},
        'PTPP.JK': {'name': 'PP (Persero) Tbk', 'tier': 'BUMN', 'market_cap': 'Medium'},
        'ADHI.JK': {'name': 'Adhi Karya Tbk', 'tier': 'BUMN', 'market_cap': 'Small'},
        'WTON.JK': {'name': 'Wijaya Karya Beton Tbk', 'tier': 'BUMN', 'market_cap': 'Small'},
        'DGIK.JK': {'name': 'Nusa Konstruksi Enjiniring Tbk', 'tier': 'Growth', 'market_cap': 'Small'}
    },
    'Retail & Consumer': {
        'MAPI.JK': {'name': 'Mitra Adiperkasa Tbk', 'tier': 'Growth', 'market_cap': 'Medium'},
        'ERAA.JK': {'name': 'Erajaya Swasembada Tbk', 'tier': 'Growth', 'market_cap': 'Small'},
        'RALS.JK': {'name': 'Ramayana Lestari Sentosa Tbk', 'tier': 'Value', 'market_cap': 'Small'},
        'LPPF.JK': {'name': 'Matahari Department Store Tbk', 'tier': 'Recovery', 'market_cap': 'Small'},
        'MPPA.JK': {'name': 'Matahari Putra Prima Tbk', 'tier': 'Recovery', 'market_cap': 'Small'}
    },
    'Otomotif': {
        'ASII.JK': {'name': 'Astra International Tbk', 'tier': 'Blue Chip', 'market_cap': 'Large'},
        'AUTO.JK': {'name': 'Astra Otoparts Tbk', 'tier': 'Blue Chip', 'market_cap': 'Medium'},
        'IMAS.JK': {'name': 'Indomobil Sukses Internasional Tbk', 'tier': 'Growth', 'market_cap': 'Medium'},
        'SMSM.JK': {'name': 'Selamat Sempurna Tbk', 'tier': 'Growth', 'market_cap': 'Small'},
        'GJTL.JK': {'name': 'Gajah Tunggal Tbk', 'tier': 'Cyclical', 'market_cap': 'Medium'}
    },
    'Teknologi': {
        'GOTO.JK': {'name': 'GoTo Gojek Tokopedia Tbk', 'tier': 'Growth', 'market_cap': 'Large'},
        'BUKA.JK': {'name': 'Bukalapak.com Tbk', 'tier': 'Growth', 'market_cap': 'Medium'},
        'MDKA.JK': {'name': 'Merdeka Copper Gold Tbk', 'tier': 'Commodity', 'market_cap': 'Medium'},
        'AMMN.JK': {'name': 'Amman Mineral Internasional Tbk', 'tier': 'Commodity', 'market_cap': 'Large'}
    },
    'Manufaktur': {
        'TPIA.JK': {'name': 'Chandra Asri Pacific Tbk', 'tier': 'Blue Chip', 'market_cap': 'Large'},
        'SRIL.JK': {'name': 'Sri Rejeki Isman Tbk', 'tier': 'Export', 'market_cap': 'Medium'},
        'ARGO.JK': {'name': 'Argo Pantes Tbk', 'tier': 'Growth', 'market_cap': 'Small'},
        'TBIG.JK': {'name': 'Tower Bersama Infrastructure Tbk', 'tier': 'Infrastructure', 'market_cap': 'Large'},
        'TOWR.JK': {'name': 'Sarana Menara Nusantara Tbk', 'tier': 'Infrastructure', 'market_cap': 'Medium'}
    }
}

class EnhancedWarrenBuffettAnalyzer:
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers
        self.model = None
        self.pca = PCA(n_components=0.95)  # For dimensionality reduction
        self.feature_names = [
            'pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_to_equity',
            'current_ratio', 'profit_margin', 'revenue_growth',
            'price_to_sma20', 'price_to_sma50', 'rsi', 'volatility',
            'price_momentum', 'volume_trend', 'earnings_growth',
            'dividend_yield', 'peg_ratio', 'price_to_sales',
            'working_capital_ratio', 'asset_turnover', 'interest_coverage',
            'beta', 'sharpe_ratio', 'max_drawdown', 'momentum_score'
        ]
        
    @st.cache_data(ttl=180, show_spinner="Fetching real-time data...")  # 3 menit cache
    def get_enhanced_stock_data(_self, symbol, period='3y'):
        """Enhanced data fetching with multiple data sources and error handling"""
        try:
            stock = yf.Ticker(symbol)
            
            # Get historical data with longer period for better analysis
            hist = stock.history(period=period, auto_adjust=True, prepost=True)
            
            if hist.empty:
                st.warning(f"No historical data available for {symbol}")
                return None
            
            # Get comprehensive info
            info = stock.info
            
            # Enhanced financial data with better error handling
            try:
                financials = stock.financials
                quarterly_financials = stock.quarterly_financials
                balance_sheet = stock.balance_sheet
                quarterly_balance_sheet = stock.quarterly_balance_sheet
                cash_flow = stock.cashflow
                quarterly_cash_flow = stock.quarterly_cashflow
                
                # Get institutional holders and insider trading
                institutional_holders = stock.institutional_holders
                insider_transactions = stock.insider_transactions
                
                # Get recommendations and analyst estimates
                recommendations = stock.recommendations
                calendar = stock.calendar
                
            except Exception as e:
                st.warning(f"Limited financial data for {symbol}: {e}")
                financials = pd.DataFrame()
                quarterly_financials = pd.DataFrame()
                balance_sheet = pd.DataFrame()
                quarterly_balance_sheet = pd.DataFrame()
                cash_flow = pd.DataFrame()
                quarterly_cash_flow = pd.DataFrame()
                institutional_holders = pd.DataFrame()
                insider_transactions = pd.DataFrame()
                recommendations = pd.DataFrame()
                calendar = pd.DataFrame()
            
            # Calculate additional technical indicators using TA library
            if len(hist) > 50:
                hist['SMA_20'] = ta.trend.sma_indicator(hist['Close'], window=20)
                hist['SMA_50'] = ta.trend.sma_indicator(hist['Close'], window=50)
                hist['SMA_200'] = ta.trend.sma_indicator(hist['Close'], window=200)
                hist['EMA_12'] = ta.trend.ema_indicator(hist['Close'], window=12)
                hist['EMA_26'] = ta.trend.ema_indicator(hist['Close'], window=26)
                
                # MACD
                hist['MACD'] = ta.trend.macd(hist['Close'])
                hist['MACD_signal'] = ta.trend.macd_signal(hist['Close'])
                hist['MACD_histogram'] = ta.trend.macd_diff(hist['Close'])
                
                # Bollinger Bands
                hist['BB_upper'] = ta.volatility.bollinger_hband(hist['Close'])
                hist['BB_lower'] = ta.volatility.bollinger_lband(hist['Close'])
                hist['BB_middle'] = ta.volatility.bollinger_mavg(hist['Close'])
                
                # RSI and Stochastic
                hist['RSI'] = ta.momentum.rsi(hist['Close'])
                hist['Stoch_K'] = ta.momentum.stoch(hist['High'], hist['Low'], hist['Close'])
                hist['Stoch_D'] = ta.momentum.stoch_signal(hist['High'], hist['Low'], hist['Close'])
                
                # Williams %R
                hist['Williams_R'] = ta.momentum.williams_r(hist['High'], hist['Low'], hist['Close'])
                
                # Volume indicators
                hist['Volume_SMA'] = ta.volume.volume_sma(hist['Close'], hist['Volume'])
                hist['MFI'] = ta.volume.money_flow_index(hist['High'], hist['Low'], hist['Close'], hist['Volume'])
                
                # Volatility indicators
                hist['ATR'] = ta.volatility.average_true_range(hist['High'], hist['Low'], hist['Close'])
                
            return {
                'history': hist,
                'info': info,
                'financials': financials,
                'quarterly_financials': quarterly_financials,
                'balance_sheet': balance_sheet,
                'quarterly_balance_sheet': quarterly_balance_sheet,
                'cash_flow': cash_flow,
                'quarterly_cash_flow': quarterly_cash_flow,
                'institutional_holders': institutional_holders,
                'insider_transactions': insider_transactions,
                'recommendations': recommendations,
                'calendar': calendar,
                'symbol': symbol,
                'last_updated': datetime.now()
            }
        except Exception as e:
            st.error(f"Critical error fetching data for {symbol}: {e}")
            return None
    
    def calculate_comprehensive_metrics(self, stock_data):
        """Calculate comprehensive metrics including advanced financial ratios"""
        try:
            info = stock_data['info']
            history = stock_data['history']
            
            metrics = {}
            
            # Enhanced Basic Metrics
            metrics['pe_ratio'] = self.safe_get(info, 'trailingPE', 0)
            metrics['forward_pe'] = self.safe_get(info, 'forwardPE', 0)
            metrics['pb_ratio'] = self.safe_get(info, 'priceToBook', 0)
            metrics['roe'] = self.safe_get(info, 'returnOnEquity', 0) * 100
            metrics['roa'] = self.safe_get(info, 'returnOnAssets', 0) * 100
            metrics['roic'] = self.calculate_roic(stock_data)
            metrics['debt_to_equity'] = self.safe_get(info, 'debtToEquity', 0)
            metrics['current_ratio'] = self.safe_get(info, 'currentRatio', 0)
            metrics['quick_ratio'] = self.safe_get(info, 'quickRatio', 0)
            metrics['dividend_yield'] = self.safe_get(info, 'dividendYield', 0) * 100
            metrics['profit_margin'] = self.safe_get(info, 'profitMargins', 0) * 100
            metrics['gross_margin'] = self.safe_get(info, 'grossMargins', 0) * 100
            metrics['operating_margin'] = self.safe_get(info, 'operatingMargins', 0) * 100
            metrics['revenue_growth'] = self.safe_get(info, 'revenueGrowth', 0) * 100
            metrics['earnings_growth'] = self.safe_get(info, 'earningsGrowth', 0) * 100
            
            # Enhanced Growth Metrics
            metrics['peg_ratio'] = self.safe_get(info, 'pegRatio', 0)
            metrics['price_to_sales'] = self.safe_get(info, 'priceToSalesTrailing12Months', 0)
            metrics['ev_to_revenue'] = self.safe_get(info, 'enterpriseToRevenue', 0)
            metrics['ev_to_ebitda'] = self.safe_get(info, 'enterpriseToEbitda', 0)
            
            # Cash Flow Metrics
            metrics['free_cash_flow'] = self.safe_get(info, 'freeCashflow', 0)
            metrics['operating_cash_flow'] = self.safe_get(info, 'operatingCashflow', 0)
            metrics['fcf_yield'] = self.calculate_fcf_yield(stock_data)
            
            # Market Metrics
            metrics['market_cap'] = self.safe_get(info, 'marketCap', 0)
            metrics['enterprise_value'] = self.safe_get(info, 'enterpriseValue', 0)
            metrics['shares_outstanding'] = self.safe_get(info, 'sharesOutstanding', 0)
            metrics['float_shares'] = self.safe_get(info, 'floatShares', 0)
            
            # Technical Analysis Metrics (Enhanced)
            if len(history) >= 200:
                current_price = history['Close'].iloc[-1]
                
                # Moving averages
                metrics['sma_20'] = history['SMA_20'].iloc[-1] if 'SMA_20' in history.columns else history['Close'].rolling(20).mean().iloc[-1]
                metrics['sma_50'] = history['SMA_50'].iloc[-1] if 'SMA_50' in history.columns else history['Close'].rolling(50).mean().iloc[-1]
                metrics['sma_200'] = history['SMA_200'].iloc[-1] if 'SMA_200' in history.columns else history['Close'].rolling(200).mean().iloc[-1]
                
                metrics['price_to_sma20'] = current_price / metrics['sma_20'] if metrics['sma_20'] > 0 else 1
                metrics['price_to_sma50'] = current_price / metrics['sma_50'] if metrics['sma_50'] > 0 else 1
                metrics['price_to_sma200'] = current_price / metrics['sma_200'] if metrics['sma_200'] > 0 else 1
                
                # Technical indicators
                metrics['rsi'] = history['RSI'].iloc[-1] if 'RSI' in history.columns else self.calculate_rsi(history['Close']).iloc[-1]
                metrics['stoch_k'] = history['Stoch_K'].iloc[-1] if 'Stoch_K' in history.columns else 50
                metrics['williams_r'] = history['Williams_R'].iloc[-1] if 'Williams_R' in history.columns else -50
                metrics['mfi'] = history['MFI'].iloc[-1] if 'MFI' in history.columns else 50
                
                # Volatility metrics
                metrics['volatility'] = history['Close'].rolling(20).std().iloc[-1]
                metrics['atr'] = history['ATR'].iloc[-1] if 'ATR' in history.columns else metrics['volatility']
                
                # Price momentum (multiple timeframes)
                metrics['price_momentum_1m'] = self.calculate_momentum(history['Close'], 22)
                metrics['price_momentum_3m'] = self.calculate_momentum(history['Close'], 66)
                metrics['price_momentum_6m'] = self.calculate_momentum(history['Close'], 132)
                metrics['price_momentum'] = (metrics['price_momentum_1m'] + metrics['price_momentum_3m']) / 2
                # Volume analysis
                volume_ma = history['Volume'].rolling(20).mean()
                metrics['volume_trend'] = history['Volume'].iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
                
                # Risk metrics
                returns = history['Close'].pct_change().dropna()
                metrics['beta'] = self.calculate_beta(returns)
                metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
                metrics['max_drawdown'] = self.calculate_max_drawdown(history['Close'])
                metrics['var_95'] = np.percentile(returns, 5) * 100  # 95% VaR
                
                # Momentum score (综合动量指标)
                momentum_factors = [
                    metrics['price_momentum_1m'],
                    metrics['price_momentum_3m'],
                    (metrics['rsi'] - 50) / 50 * 100,  # RSI normalized
                    (current_price / metrics['sma_20'] - 1) * 100
                ]
                metrics['momentum_score'] = np.mean(momentum_factors)
                
            else:
                # Default values for stocks with limited data
                metrics.update({
                    'sma_20': history['Close'].iloc[-1],
                    'sma_50': history['Close'].iloc[-1],
                    'sma_200': history['Close'].iloc[-1],
                    'price_to_sma20': 1,
                    'price_to_sma50': 1,
                    'price_to_sma200': 1,
                    'rsi': 50,
                    'stoch_k': 50,
                    'williams_r': -50,
                    'mfi': 50,
                    'volatility': 0,
                    'atr': 0,
                    'price_momentum_1m': 0,
                    'price_momentum_3m': 0,
                    'price_momentum_6m': 0,
                    'price_momentum': 0,
                    'volume_trend': 1,
                    'beta': 1,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'var_95': 0,
                    'momentum_score': 0
                })
            
            # Additional computed metrics
            metrics['working_capital_ratio'] = self.calculate_working_capital_ratio(stock_data)
            metrics['asset_turnover'] = self.calculate_asset_turnover(stock_data)
            metrics['interest_coverage'] = self.calculate_interest_coverage(stock_data)
            
            # Quality score components
            metrics['consistency_score'] = self.calculate_consistency_score(stock_data)
            metrics['growth_quality'] = self.calculate_growth_quality(stock_data)
            metrics['management_efficiency'] = self.calculate_management_efficiency(stock_data)
            
            return metrics
        except Exception as e:
            st.error(f"Error calculating comprehensive metrics for {stock_data.get('symbol', 'Unknown')}: {e}")
            return self.get_default_metrics()
    
    def safe_get(self, dictionary, key, default=0):
        """Safely get value from dictionary with proper type checking"""
        try:
            value = dictionary.get(key, default)
            if value is None or (isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value))):
                return default
            return float(value) if isinstance(value, (int, float)) else default
        except:
            return default
    
    def calculate_momentum(self, prices, periods):
        """Calculate price momentum over specified periods"""
        try:
            if len(prices) >= periods:
                return (prices.iloc[-1] / prices.iloc[-periods] - 1) * 100
            return 0
        except:
            return 0
    
    def calculate_beta(self, stock_returns, market_returns=None):
        """Calculate beta (systematic risk)"""
        try:
            if market_returns is None:
                # Use a simple market proxy or default
                return 1.0
            
            if len(stock_returns) == len(market_returns) and len(stock_returns) > 30:
                covariance = np.cov(stock_returns, market_returns)[0][1]
                market_variance = np.var(market_returns)
                return covariance / market_variance if market_variance != 0 else 1.0
            return 1.0
        except:
            return 1.0
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.06):
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 30:
                return 0
            
            annual_return = returns.mean() * 252
            annual_std = returns.std() * np.sqrt(252)
            
            return (annual_return - risk_free_rate) / annual_std if annual_std != 0 else 0
        except:
            return 0
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        try:
            rolling_max = prices.expanding().max()
            drawdown = (prices - rolling_max) / rolling_max
            return abs(drawdown.min()) * 100
        except:
            return 0
    
    def calculate_roic(self, stock_data):
        """Calculate Return on Invested Capital"""
        try:
            info = stock_data['info']
            total_revenue = self.safe_get(info, 'totalRevenue', 0)
            operating_income = total_revenue * (self.safe_get(info, 'operatingMargins', 0))
            total_debt = self.safe_get(info, 'totalDebt', 0)
            total_equity = self.safe_get(info, 'totalStockholderEquity', 0)
            
            invested_capital = total_debt + total_equity
            
            if invested_capital > 0:
                return (operating_income / invested_capital) * 100
            return 0
        except:
            return 0
    
    def calculate_working_capital_ratio(self, stock_data):
        """Calculate working capital efficiency"""
        try:
            info = stock_data['info']
            current_assets = self.safe_get(info, 'totalCurrentAssets', 0)
            current_liabilities = self.safe_get(info, 'totalCurrentLiabilities', 0)
            
            if current_liabilities > 0:
                return (current_assets - current_liabilities) / current_liabilities
            return 0
        except:
            return 0
    
    def calculate_asset_turnover(self, stock_data):
        """Calculate asset turnover ratio"""
        try:
            info = stock_data['info']
            revenue = self.safe_get(info, 'totalRevenue', 0)
            total_assets = self.safe_get(info, 'totalAssets', 0)
            
            if total_assets > 0:
                return revenue / total_assets
            return 0
        except:
            return 0
    
    def calculate_interest_coverage(self, stock_data):
        """Calculate interest coverage ratio"""
        try:
            info = stock_data['info']
            ebit = self.safe_get(info, 'ebitda', 0) - self.safe_get(info, 'totalRevenue', 0) * 0.1  # Rough estimate
            interest_expense = self.safe_get(info, 'interestExpense', 0)
            
            if interest_expense > 0:
                return abs(ebit / interest_expense)
            return 999  # No debt scenario
        except:
            return 0
    
    def calculate_fcf_yield(self, stock_data):
        """Calculate Free Cash Flow Yield"""
        try:
            info = stock_data['info']
            fcf = self.safe_get(info, 'freeCashflow', 0)
            market_cap = self.safe_get(info, 'marketCap', 0)
            
            if market_cap > 0:
                return (fcf / market_cap) * 100
            return 0
        except:
            return 0
    
    def calculate_consistency_score(self, stock_data):
        """Calculate earnings consistency score"""
        try:
            # This would analyze quarterly earnings consistency
            # For now, return a placeholder based on available data
            info = stock_data['info']
            revenue_growth = self.safe_get(info, 'revenueGrowth', 0)
            earnings_growth = self.safe_get(info, 'earningsGrowth', 0)
            
            # Simple consistency metric based on growth stability
            if abs(revenue_growth - earnings_growth) < 0.1:  # Similar growth rates
                return 80
            elif abs(revenue_growth - earnings_growth) < 0.2:
                return 60
            else:
                return 40
        except:
            return 50
    
    def calculate_growth_quality(self, stock_data):
        """Assess quality of growth"""
        try:
            info = stock_data['info']
            revenue_growth = self.safe_get(info, 'revenueGrowth', 0)
            earnings_growth = self.safe_get(info, 'earningsGrowth', 0)
            fcf = self.safe_get(info, 'freeCashflow', 0)
            
            quality_score = 0
            
            # Sustainable growth indicators
            if revenue_growth > 0 and earnings_growth > 0:
                quality_score += 30
            
            if fcf > 0:
                quality_score += 25
            
            if revenue_growth > 0 and revenue_growth < 0.5:  # Moderate, sustainable growth
                quality_score += 25
            
            if self.safe_get(info, 'operatingMargins', 0) > 0.1:
                quality_score += 20
            
            return quality_score
        except:
            return 50
    
    def calculate_management_efficiency(self, stock_data):
        """Calculate management efficiency score"""
        try:
            info = stock_data['info']
            
            efficiency_score = 0
            
            # ROE and ROA
            roe = self.safe_get(info, 'returnOnEquity', 0)
            roa = self.safe_get(info, 'returnOnAssets', 0)
            
            if roe > 0.15:
                efficiency_score += 30
            elif roe > 0.10:
                efficiency_score += 20
            
            if roa > 0.10:
                efficiency_score += 25
            elif roa > 0.05:
                efficiency_score += 15
            
            # Asset turnover
            asset_turnover = self.calculate_asset_turnover(stock_data)
            if asset_turnover > 1.0:
                efficiency_score += 25
            elif asset_turnover > 0.5:
                efficiency_score += 15
            
            # Debt management
            debt_ratio = self.safe_get(info, 'debtToEquity', 0)
            if debt_ratio < 0.3:
                efficiency_score += 20
            elif debt_ratio < 0.5:
                efficiency_score += 10
            
            return efficiency_score
        except:
            return 50
    
    def get_default_metrics(self):
        """Return default metrics when data is unavailable"""
        return {key: 0 for key in self.feature_names}
    
    def calculate_rsi(self, prices, period=14):
        """Enhanced RSI calculation"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def enhanced_buffett_score(self, metrics):
        """Enhanced Warren Buffett scoring system (0-100) with weighted categories"""
        score = 0
        weights = {
            'value': 0.25,      # 25% - Value metrics
            'quality': 0.30,    # 30% - Quality metrics  
            'growth': 0.20,     # 20% - Growth metrics
            'financial': 0.15,  # 15% - Financial health
            'management': 0.10  # 10% - Management efficiency
        }
        
        # Value Score (0-100)
        value_score = 0
        pe_ratio = metrics.get('pe_ratio', 999)
        pb_ratio = metrics.get('pb_ratio', 999)
        peg_ratio = metrics.get('peg_ratio', 999)
        
        # P/E scoring
        if 0 < pe_ratio < 8:
            value_score += 40
        elif pe_ratio < 12:
            value_score += 35
        elif pe_ratio < 15:
            value_score += 25
        elif pe_ratio < 20:
            value_score += 15
        elif pe_ratio < 25:
            value_score += 5
        
        # P/B scoring
        if 0 < pb_ratio < 0.8:
            value_score += 30
        elif pb_ratio < 1.2:
            value_score += 25
        elif pb_ratio < 1.5:
            value_score += 20
        elif pb_ratio < 2.0:
            value_score += 10
        elif pb_ratio < 3.0:
            value_score += 5
        
        # PEG scoring
        if 0 < peg_ratio < 0.8:
            value_score += 30
        elif peg_ratio < 1.0:
            value_score += 25
        elif peg_ratio < 1.5:
            value_score += 15
        elif peg_ratio < 2.0:
            value_score += 5
        
        # Quality Score (0-100)
        quality_score = 0
        roe = metrics.get('roe', 0)
        roa = metrics.get('roa', 0)
        roic = metrics.get('roic', 0)
        profit_margin = metrics.get('profit_margin', 0)
        
        # ROE scoring
        if roe > 25:
            quality_score += 30
        elif roe > 20:
            quality_score += 25
        elif roe > 15:
            quality_score += 20
        elif roe > 10:
            quality_score += 10
        elif roe > 5:
            quality_score += 5
        
        # ROA scoring
        if roa > 15:
            quality_score += 25
        elif roa > 10:
            quality_score += 20
        elif roa > 7:
            quality_score += 15
        elif roa > 5:
            quality_score += 10
        
        # ROIC scoring
        if roic > 20:
            quality_score += 25
        elif roic > 15:
            quality_score += 20
        elif roic > 10:
            quality_score += 10
        
        # Profit margin scoring
        if profit_margin > 20:
            quality_score += 20
        elif profit_margin > 15:
            quality_score += 15
        elif profit_margin > 10:
            quality_score += 10
        elif profit_margin > 5:
            quality_score += 5
        
        # Growth Score (0-100)
        growth_score = 0
        revenue_growth = metrics.get('revenue_growth', 0)
        earnings_growth = metrics.get('earnings_growth', 0)
        consistency = metrics.get('consistency_score', 50)
        
        # Revenue growth scoring
        if 5 <= revenue_growth <= 25:
            growth_score += 35
        elif 0 <= revenue_growth < 5:
            growth_score += 20
        elif revenue_growth > 25:
            growth_score += 15  # Too high might be unsustainable
        
        # Earnings growth scoring
        if 5 <= earnings_growth <= 20:
            growth_score += 35
        elif 0 <= earnings_growth < 5:
            growth_score += 20
        elif earnings_growth > 20:
            growth_score += 15
        
        # Consistency bonus
        growth_score += consistency * 0.3
        
        # Financial Health Score (0-100)
        financial_score = 0
        debt_ratio = metrics.get('debt_to_equity', 999)
        current_ratio = metrics.get('current_ratio', 0)
        interest_coverage = metrics.get('interest_coverage', 0)
        
        # Debt scoring
        if debt_ratio < 10:
            financial_score += 40
        elif debt_ratio < 20:
            financial_score += 35
        elif debt_ratio < 30:
            financial_score += 25
        elif debt_ratio < 50:
            financial_score += 15
        elif debt_ratio < 70:
            financial_score += 5
        
        # Liquidity scoring
        if current_ratio > 2.5:
            financial_score += 30
        elif current_ratio > 2.0:
            financial_score += 25
        elif current_ratio > 1.5:
            financial_score += 20
        elif current_ratio > 1.2:
            financial_score += 10
        
        # Interest coverage
        if interest_coverage > 10:
            financial_score += 30
        elif interest_coverage > 5:
            financial_score += 20
        elif interest_coverage > 2:
            financial_score += 10
        
        # Management Score (0-100)
        management_score = metrics.get('management_efficiency', 50)
        
        # Calculate weighted final score
        final_score = (
            value_score * weights['value'] +
            quality_score * weights['quality'] +
            growth_score * weights['growth'] +
            financial_score * weights['financial'] +
            management_score * weights['management']
        )
        
        return {
            'total_score': min(int(final_score), 100),
            'value_score': min(int(value_score), 100),
            'quality_score': min(int(quality_score), 100),
            'growth_score': min(int(growth_score), 100),
            'financial_score': min(int(financial_score), 100),
            'management_score': min(int(management_score), 100)
        }
    
    def multi_method_intrinsic_value(self, stock_data):
        """Advanced intrinsic value calculation with multiple sophisticated methods"""
        try:
            info = stock_data['info']
            history = stock_data['history']
            methods = {}
            
            # Method 1: Enhanced DCF with multiple scenarios
            methods['dcf_conservative'] = self.enhanced_dcf_valuation(stock_data, 'conservative')
            methods['dcf_moderate'] = self.enhanced_dcf_valuation(stock_data, 'moderate')
            methods['dcf_optimistic'] = self.enhanced_dcf_valuation(stock_data, 'optimistic')
            
            # Method 2: Enhanced Graham Formula
            methods['graham_enhanced'] = self.enhanced_graham_number(stock_data)
            
            # Method 3: Dividend Discount Model
            methods['ddm'] = self.dividend_discount_model(stock_data)
            
            # Method 4: Asset-based valuation
            methods['asset_based'] = self.asset_based_valuation(stock_data)
            
            # Method 5: Relative valuation
            methods['relative_valuation'] = self.relative_valuation(stock_data)
            
            # Method 6: Sum of Parts (if applicable)
            methods['sum_of_parts'] = self.sum_of_parts_valuation(stock_data)
            
            # Calculate weighted average intrinsic value
            valid_methods = {k: v for k, v in methods.items() if v and v > 0}
            
            if valid_methods:
                # Weight different methods based on reliability
                method_weights = {
                    'dcf_moderate': 0.25,
                    'dcf_conservative': 0.20,
                    'graham_enhanced': 0.20,
                    'ddm': 0.15,
                    'relative_valuation': 0.10,
                    'asset_based': 0.05,
                    'dcf_optimistic': 0.05
                }
                
                weighted_sum = 0
                total_weight = 0
                
                for method, value in valid_methods.items():
                    weight = method_weights.get(method, 0.05)
                    weighted_sum += value * weight
                    total_weight += weight
                
                methods['weighted_average'] = weighted_sum / total_weight if total_weight > 0 else 0
            
            return methods
        except Exception as e:
            st.error(f"Error calculating intrinsic values: {e}")
            return {}
    
    def enhanced_dcf_valuation(self, stock_data, scenario='moderate'):
        """Enhanced DCF with scenario analysis"""
        try:
            info = stock_data['info']
            fcf = self.safe_get(info, 'freeCashflow', 0)
            shares = self.safe_get(info, 'sharesOutstanding', 0)
            
            if not fcf or not shares or fcf <= 0:
                return None
            
            # Scenario-based assumptions
            scenarios = {
                'conservative': {'growth': 0.03, 'terminal': 0.02, 'discount': 0.12},
                'moderate': {'growth': 0.06, 'terminal': 0.025, 'discount': 0.10},
                'optimistic': {'growth': 0.10, 'terminal': 0.03, 'discount': 0.08}
            }
            
            params = scenarios[scenario]
            growth_rate = min(params['growth'], max(0.01, self.safe_get(info, 'revenueGrowth', 0.03)))
            
            # 10-year explicit forecast
            cash_flows = []
            for year in range(1, 11):
                # Declining growth rate over time
                adjusted_growth = growth_rate * (0.9 ** (year - 1))
                future_cf = fcf * ((1 + adjusted_growth) ** year)
                present_value = future_cf / ((1 + params['discount']) ** year)
                cash_flows.append(present_value)
            
            # Terminal value with Gordon Growth Model
            terminal_cf = cash_flows[-1] * (1 + params['terminal'])
            terminal_value = terminal_cf / (params['discount'] - params['terminal'])
            terminal_pv = terminal_value / ((1 + params['discount']) ** 10)
            
            total_enterprise_value = sum(cash_flows) + terminal_pv
            
            # Adjust for net cash/debt
            net_debt = self.safe_get(info, 'totalDebt', 0) - self.safe_get(info, 'totalCash', 0)
            equity_value = total_enterprise_value - net_debt
            
            intrinsic_value_per_share = equity_value / shares
            
            return max(intrinsic_value_per_share, 0)
        except:
            return None
    
    def enhanced_graham_number(self, stock_data):
        """Enhanced Benjamin Graham Number with modifications"""
        try:
            info = stock_data['info']
            eps = self.safe_get(info, 'trailingEps', 0)
            book_value = self.safe_get(info, 'bookValue', 0)
            
            if eps > 0 and book_value > 0:
                # Original Graham: sqrt(22.5 * EPS * BVPS)
                # Enhanced: adjust for current market conditions
                market_adjustment = 0.8  # Conservative adjustment for current markets
                return (22.5 * eps * book_value * market_adjustment) ** 0.5
            return None
        except:
            return None
    
    def dividend_discount_model(self, stock_data):
        """Dividend Discount Model for dividend-paying stocks"""
        try:
            info = stock_data['info']
            dividend_rate = self.safe_get(info, 'dividendRate', 0)
            dividend_yield = self.safe_get(info, 'dividendYield', 0)
            
            if dividend_rate <= 0:
                return None
            
            # Estimate dividend growth rate
            payout_ratio = self.safe_get(info, 'payoutRatio', 0.4)
            roe = self.safe_get(info, 'returnOnEquity', 0.1)
            sustainable_growth = roe * (1 - payout_ratio)
            
            # Conservative discount rate
            required_return = 0.12
            
            if sustainable_growth < required_return:
                # Gordon Growth Model
                return dividend_rate / (required_return - sustainable_growth)
            
            return None
        except:
            return None
    
    def asset_based_valuation(self, stock_data):
        """Asset-based valuation approach"""
        try:
            info = stock_data['info']
            book_value = self.safe_get(info, 'bookValue', 0)
            
            # Adjust book value for asset quality
            roe = self.safe_get(info, 'returnOnEquity', 0)
            
            # If ROE is high, assets are productive
            if roe > 0.15:
                adjustment_factor = 1.2
            elif roe > 0.10:
                adjustment_factor = 1.1
            elif roe > 0.05:
                adjustment_factor = 1.0
            else:
                adjustment_factor = 0.8
            
            return book_value * adjustment_factor
        except:
            return None
    
    def relative_valuation(self, stock_data):
        """Relative valuation using industry multiples"""
        try:
            info = stock_data['info']
            eps = self.safe_get(info, 'trailingEps', 0)
            
            # Industry average P/E (simplified - would need sector data in reality)
            sector_pe_avg = {
                'Banking': 12, 'Consumer': 18, 'Telecom': 15, 'Mining': 10,
                'Property': 14, 'Energy': 12, 'Infrastructure': 16, 
                'Retail': 20, 'Automotive': 15, 'Technology': 25, 'Manufacturing': 16
            }
            
            # Default to market average if sector unknown
            estimated_fair_pe = 15
            
            if eps > 0:
                return eps * estimated_fair_pe
            return None
        except:
            return None
    
    def sum_of_parts_valuation(self, stock_data):
        """Sum of parts valuation for conglomerates"""
        try:
            # This would be implemented for complex holding companies
            # For now, return None as it requires detailed segment data
            return None
        except:
            return None
    
    def prepare_enhanced_ml_features(self, stock_data):
        """Prepare enhanced feature set for machine learning"""
        metrics = self.calculate_comprehensive_metrics(stock_data)
        
        if not metrics:
            return None
        
        # Extract features matching our enhanced feature set
        features = []
        for feature_name in self.feature_names:
            value = metrics.get(feature_name, 0)
            # Handle any remaining NaN or infinite values
            if np.isnan(value) or np.isinf(value):
                value = 0
            features.append(value)
        
        return np.array(features)
    
    def train_advanced_ensemble_model(self, all_stock_data):
        """Train advanced ensemble model with cross-validation and feature selection"""
        try:
            features_list = []
            targets_short = []  # 1-month returns
            targets_medium = []  # 3-month returns
            targets_long = []   # 6-month returns
            symbols = []
            
            min_history_length = 200  # Require more data for robust training
            
            for symbol, data in all_stock_data.items():
                if data is None or len(data['history']) < min_history_length:
                    continue
                
                features = self.prepare_enhanced_ml_features(data)
                if features is None or len(features) != len(self.feature_names):
                    continue
                
                history = data['history']
                
                # Multiple prediction horizons
                try:
                    current_price = history['Close'].iloc[-150]  # Use historical data for training
                    price_1m = history['Close'].iloc[-130]
                    price_3m = history['Close'].iloc[-90] 
                    price_6m = history['Close'].iloc[-30]
                    
                    # Calculate returns
                    return_1m = (price_1m - current_price) / current_price
                    return_3m = (price_3m - current_price) / current_price  
                    return_6m = (price_6m - current_price) / current_price
                    
                    features_list.append(features)
                    targets_short.append(return_1m)
                    targets_medium.append(return_3m)
                    targets_long.append(return_6m)
                    symbols.append(symbol)
                    
                except IndexError:
                    continue
            
            if len(features_list) < 20:  # Need minimum samples
                st.warning("Insufficient data for machine learning model training")
                return None
            
            X = np.array(features_list)
            y_short = np.array(targets_short)
            y_medium = np.array(targets_medium)
            y_long = np.array(targets_long)
            
            # Remove outliers using IQR method
            def remove_outliers(X, y):
                q1, q3 = np.percentile(y, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 2.0 * iqr  # More conservative
                upper_bound = q3 + 2.0 * iqr
                
                mask = (y >= lower_bound) & (y <= upper_bound)
                return X[mask], y[mask]
            
            X_clean, y_short_clean = remove_outliers(X, y_short)
            _, y_medium_clean = remove_outliers(X, y_medium)
            _, y_long_clean = remove_outliers(X, y_long)
            
            # Feature scaling with robust scaler
            self.scaler.fit(X_clean)
            X_scaled = self.scaler.transform(X_clean)
            
            # Feature selection using correlation and importance
            feature_selector = self.select_best_features(X_scaled, y_medium_clean)
            X_selected = X_scaled[:, feature_selector]
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Multiple model ensemble
            models = {
                'rf': RandomForestRegressor(
                    n_estimators=200, 
                    max_depth=12, 
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'gb': GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    min_samples_split=5,
                    random_state=42
                ),
                'linear': LinearRegression()
            }
            
            # Train models for different timeframes
            trained_models = {}
            cv_scores = {}
            
            for timeframe, y_target in [('1m', y_short_clean), ('3m', y_medium_clean), ('6m', y_long_clean)]:
                if len(y_target) != len(X_selected):
                    continue
                    
                timeframe_models = {}
                timeframe_scores = {}
                
                for name, model in models.items():
                    # Cross-validation
                    cv_score = cross_val_score(model, X_selected, y_target, cv=tscv, scoring='r2')
                    timeframe_scores[name] = cv_score.mean()
                    
                    # Train on full dataset
                    model.fit(X_selected, y_target)
                    timeframe_models[name] = model
                
                trained_models[timeframe] = timeframe_models
                cv_scores[timeframe] = timeframe_scores
            
            # Create voting ensemble for best timeframe (3m)
            if '3m' in trained_models:
                voting_regressor = VotingRegressor([
                    ('rf', trained_models['3m']['rf']),
                    ('gb', trained_models['3m']['gb']),
                    ('linear', trained_models['3m']['linear'])
                ])
                voting_regressor.fit(X_selected, y_medium_clean)
                trained_models['3m']['ensemble'] = voting_regressor
            
            self.model = trained_models
            self.feature_selector = feature_selector
            
            # Model evaluation
            evaluation_results = {
                'cv_scores': cv_scores,
                'n_samples': len(X_clean),
                'n_features_selected': len(feature_selector),
                'feature_importance': self.get_feature_importance(trained_models.get('3m', {}).get('rf')),
                'model_performance': self.evaluate_model_performance(X_selected, y_medium_clean, trained_models.get('3m', {}))
            }
            
            return evaluation_results
            
        except Exception as e:
            st.error(f"Error training ML model: {e}")
            return None
    
    def select_best_features(self, X, y, top_k=15):
        """Select best features using mutual information and correlation"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X, y)
            
            # Get top features
            top_features = np.argsort(mi_scores)[-top_k:]
            
            return top_features
        except:
            # Return all features if selection fails
            return np.arange(len(self.feature_names))
    
    def get_feature_importance(self, model):
        """Get feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                selected_features = [self.feature_names[i] for i in self.feature_selector]
                importance_df = pd.DataFrame({
                    'feature': selected_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                return importance_df
        except:
            pass
        return pd.DataFrame()
    
    def evaluate_model_performance(self, X, y, models):
        """Evaluate model performance with multiple metrics"""
        try:
            performance = {}
            
            for name, model in models.items():
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X)
                    
                    performance[name] = {
                        'r2': r2_score(y, y_pred),
                        'mse': mean_squared_error(y, y_pred),
                        'mae': mean_absolute_error(y, y_pred),
                        'std_error': np.std(y - y_pred)
                    }
            
            return performance
        except:
            return {}
    
    def predict_enhanced_returns(self, stock_data, timeframe='3m'):
        """Enhanced return prediction with confidence intervals"""
        if not self.model or timeframe not in self.model:
            return None
        
        features = self.prepare_enhanced_ml_features(stock_data)
        if features is None:
            return None
        
        try:
            # Scale and select features
            X = features.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            X_selected = X_scaled[:, self.feature_selector]
            
            models = self.model[timeframe]
            predictions = {}
            
            # Get predictions from all models
            for name, model in models.items():
                if hasattr(model, 'predict'):
                    pred = model.predict(X_selected)[0]
                    predictions[name] = pred
            
            # Ensemble prediction
            ensemble_pred = np.mean(list(predictions.values()))
            prediction_std = np.std(list(predictions.values()))
            
            # Confidence intervals (approximate)
            confidence_95 = {
                'lower': ensemble_pred - 1.96 * prediction_std,
                'upper': ensemble_pred + 1.96 * prediction_std
            }
            
            return {
                'prediction': ensemble_pred,
                'confidence_interval': confidence_95,
                'individual_predictions': predictions,
                'uncertainty': prediction_std
            }
            
        except Exception as e:
            st.error(f"Error in ML prediction: {e}")
            return None
    
    def enhanced_investment_recommendation(self, metrics, intrinsic_values, current_price, ml_prediction, stock_info):
        """Enhanced investment recommendation with risk assessment"""
        try:
            # Calculate Buffett scores
            scores = self.enhanced_buffett_score(metrics)
            total_score = scores['total_score']
            
            # Calculate margin of safety from multiple methods
            margins = []
            valid_methods = []
            
            for method, value in intrinsic_values.items():
                if value and value > 0 and method != 'weighted_average':
                    margin = (value - current_price) / value * 100
                    margins.append(margin)
                    valid_methods.append(method)
            
            avg_margin = np.mean(margins) if margins else 0
            margin_consistency = 100 - np.std(margins) if len(margins) > 1 else 0
            
            # ML prediction analysis
            ml_return = 0
            ml_confidence = 0
            if ml_prediction:
                ml_return = ml_prediction['prediction'] * 100
                ml_confidence = max(0, 100 - ml_prediction['uncertainty'] * 1000)
            
            # Risk assessment
            risk_score, risk_factors = self.comprehensive_risk_assessment(metrics, stock_info)
            
            # Enhanced recommendation logic
            recommendation_score = (
                total_score * 0.4 +           # 40% Buffett fundamentals
                max(0, avg_margin) * 0.3 +    # 30% Margin of safety
                ml_confidence * 0.2 +         # 20% ML confidence
                (100 - risk_score) * 0.1      # 10% Risk-adjusted
            )
            
            # Determine recommendation
            if (recommendation_score >= 85 and total_score >= 75 and 
                avg_margin > 25 and ml_return > 8 and risk_score < 30):
                return "🚀 STRONG BUY", self.generate_recommendation_reason(
                    "Exceptional opportunity", total_score, avg_margin, ml_return, risk_score
                ), recommendation_score
                
            elif (recommendation_score >= 70 and total_score >= 65 and 
                  avg_margin > 15 and risk_score < 50):
                return "✅ BUY", self.generate_recommendation_reason(
                    "Attractive investment", total_score, avg_margin, ml_return, risk_score
                ), recommendation_score
                
            elif (recommendation_score >= 55 and total_score >= 50 and 
                  avg_margin > 5):
                return "⏸️ HOLD", self.generate_recommendation_reason(
                    "Fair value range", total_score, avg_margin, ml_return, risk_score
                ), recommendation_score
                
            elif recommendation_score < 40 or risk_score > 70 or avg_margin < -25:
                return "📉 SELL", self.generate_recommendation_reason(
                    "High risk or overvalued", total_score, avg_margin, ml_return, risk_score
                ), recommendation_score
            else:
                return "⚠️ WATCH", self.generate_recommendation_reason(
                    "Mixed signals", total_score, avg_margin, ml_return, risk_score
                ), recommendation_score
                
        except Exception as e:
            st.error(f"Error in recommendation generation: {e}")
            return "❓ UNKNOWN", "Unable to generate recommendation", 0
    
    def generate_recommendation_reason(self, base_reason, buffett_score, margin, ml_return, risk_score):
        """Generate detailed recommendation reasoning"""
        reasons = [base_reason]
        
        if buffett_score >= 80:
            reasons.append("excellent fundamentals")
        elif buffett_score >= 65:
            reasons.append("solid fundamentals")
        elif buffett_score < 40:
            reasons.append("weak fundamentals")
        
        if margin > 30:
            reasons.append("high safety margin")
        elif margin > 15:
            reasons.append("adequate safety margin") 
        elif margin < 0:
            reasons.append("overvalued")
        
        if abs(ml_return) > 10:
            reasons.append(f"strong ML signal ({ml_return:+.1f}%)")
        elif abs(ml_return) > 5:
            reasons.append(f"moderate ML signal ({ml_return:+.1f}%)")
        
        if risk_score > 60:
            reasons.append("high risk profile")
        elif risk_score < 30:
            reasons.append("low risk profile")
        
        return " + ".join(reasons[:4])  # Limit to 4 main reasons
    
    def comprehensive_risk_assessment(self, metrics, stock_info):
        """Comprehensive risk assessment"""
        risk_score = 0
        risk_factors = []
        
        try:
            # Financial risk (0-40 points)
            debt_ratio = metrics.get('debt_to_equity', 0)
            if debt_ratio > 80:
                risk_score += 25
                risk_factors.append("Very high debt levels")
            elif debt_ratio > 50:
                risk_score += 15
                risk_factors.append("High debt levels")
            elif debt_ratio > 30:
                risk_score += 8
                risk_factors.append("Moderate debt levels")
            
            current_ratio = metrics.get('current_ratio', 0)
            if current_ratio < 1.0:
                risk_score += 15
                risk_factors.append("Liquidity concerns")
            elif current_ratio < 1.2:
                risk_score += 8
                risk_factors.append("Tight liquidity")
            
            # Valuation risk (0-30 points)
            pe_ratio = metrics.get('pe_ratio', 0)
            if pe_ratio > 40:
                risk_score += 20
                risk_factors.append("Extreme overvaluation")
            elif pe_ratio > 25:
                risk_score += 10
                risk_factors.append("High valuation")
            
            # Market risk (0-20 points)
            beta = metrics.get('beta', 1)
            if beta > 1.5:
                risk_score += 15
                risk_factors.append("High market sensitivity")
            elif beta > 1.2:
                risk_score += 8
                risk_factors.append("Above-average volatility")
            
            # Volatility risk (0-10 points)
            volatility = metrics.get('volatility', 0)
            max_drawdown = metrics.get('max_drawdown', 0)
            if max_drawdown > 40:
                risk_score += 10
                risk_factors.append("High historical drawdowns")
            elif max_drawdown > 25:
                risk_score += 5
                risk_factors.append("Moderate volatility")
            
            return min(risk_score, 100), risk_factors
            
        except Exception as e:
            return 50, ["Unable to assess risk factors"]

@st.cache_data(ttl=900)  # 15 menit cache untuk loading multiple stocks
def load_enhanced_stock_data(selected_stocks, progress_bar=None):
    """Enhanced parallel data loading with progress tracking"""
    analyzer = EnhancedWarrenBuffettAnalyzer()
    
    def load_single_stock(symbol):
        try:
            return symbol, analyzer.get_enhanced_stock_data(symbol, '3y')
        except Exception as e:
            st.warning(f"Failed to load {symbol}: {e}")
            return symbol, None
    
    all_data = {}
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_symbol = {executor.submit(load_single_stock, symbol): symbol 
                           for symbol in selected_stocks}
        
        completed = 0
        for future in as_completed(future_to_symbol):
            symbol, data = future.result()
            if data:
                all_data[symbol] = data
            
            completed += 1
            if progress_bar:
                progress_bar.progress(completed / len(selected_stocks))
    
    return all_data

def create_enhanced_visualizations(stock_data, metrics, analyzer):
    """Create enhanced, interactive visualizations"""
    
    # 1. Advanced Candlestick Chart with Technical Indicators
    history = stock_data['history']
    
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Price Action & Technical Indicators', 'Volume & Money Flow',
            'RSI & Stochastic Oscillator', 'MACD Analysis', 
            'Bollinger Bands & Volatility', 'Fundamental vs Technical Score'
        ),
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"colspan": 2}, None],
            [{"type": "polar"}, {"secondary_y": False}]
        ],
        vertical_spacing=0.06,
        row_heights=[0.35, 0.25, 0.25, 0.15]
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=history.index,
            open=history['Open'],
            high=history['High'], 
            low=history['Low'],
            close=history['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4757'
        ), row=1, col=1
    )
    
    # Moving averages with different styles
    if 'SMA_20' in history.columns:
        fig.add_trace(
            go.Scatter(x=history.index, y=history['SMA_20'], 
                      name='SMA 20', line=dict(color='orange', width=2)),
            row=1, col=1
        )
    
    if 'SMA_50' in history.columns:
        fig.add_trace(
            go.Scatter(x=history.index, y=history['SMA_50'],
                      name='SMA 50', line=dict(color='red', width=2)),
            row=1, col=1
        )
    
    if 'SMA_200' in history.columns:
        fig.add_trace(
            go.Scatter(x=history.index, y=history['SMA_200'],
                      name='SMA 200', line=dict(color='purple', width=3)),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'BB_upper' in history.columns:
        fig.add_trace(
            go.Scatter(x=history.index, y=history['BB_upper'],
                      name='BB Upper', line=dict(color='gray', dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=history.index, y=history['BB_lower'],
                      name='BB Lower', line=dict(color='gray', dash='dot'),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
    
    # Volume analysis
    colors = ['green' if close >= open else 'red' 
              for close, open in zip(history['Close'], history['Open'])]
    
    fig.add_trace(
        go.Bar(x=history.index, y=history['Volume'], 
               name='Volume', marker_color=colors, opacity=0.6),
        row=1, col=2
    )
    
    # Money Flow Index (if available)
    if 'MFI' in history.columns:
        fig.add_trace(
            go.Scatter(x=history.index, y=history['MFI'],
                      name='Money Flow Index', line=dict(color='blue')),
            row=1, col=2, secondary_y=True
        )
    
    # RSI and Stochastic
    if 'RSI' in history.columns:
        fig.add_trace(
            go.Scatter(x=history.index, y=history['RSI'],
                      name='RSI', line=dict(color='purple', width=2)),
            row=2, col=1
        )
    
    if 'Stoch_K' in history.columns:
        fig.add_trace(
            go.Scatter(x=history.index, y=history['Stoch_K'],
                      name='Stoch %K', line=dict(color='orange')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=history.index, y=history['Stoch_D'],
                      name='Stoch %D', line=dict(color='red')),
            row=2, col=1
        )
    
    # Add horizontal lines for RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="orange", row=2, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="orange", row=2, col=1)
    
    # MACD
    if 'MACD' in history.columns:
        fig.add_trace(
            go.Scatter(x=history.index, y=history['MACD'],
                      name='MACD', line=dict(color='blue')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=history.index, y=history['MACD_signal'],
                      name='Signal', line=dict(color='red')),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(x=history.index, y=history['MACD_histogram'],
                   name='Histogram', marker_color='gray', opacity=0.6),
            row=2, col=2
        )
    
    # Bollinger Bands detailed view
    if 'BB_upper' in history.columns:
        recent_data = history.tail(100)  # Last 100 days
        
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['Close'],
                      name='Price (Recent)', line=dict(color='black', width=3)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['BB_upper'],
                      name='Upper Band', line=dict(color='red', dash='dash')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['BB_lower'],
                      name='Lower Band', line=dict(color='green', dash='dash')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['BB_middle'],
                      name='Middle Band', line=dict(color='blue')),
            row=3, col=1
        )
    
    # Fundamental Radar Chart
    scores = analyzer.enhanced_buffett_score(metrics)
    
    radar_data = {
        'Value': scores['value_score'],
        'Quality': scores['quality_score'], 
        'Growth': scores['growth_score'],
        'Financial Health': scores['financial_score'],
        'Management': scores['management_score']
    }
    
    fig.add_trace(
        go.Scatterpolar(
            r=list(radar_data.values()),
            theta=list(radar_data.keys()),
            fill='toself',
            name='Fundamental Score',
            line_color='blue',
            fillcolor='rgba(0,100,255,0.25)'
        ), row=4, col=1
    )
    
    # Risk-Return Scatter (last 60 days)
    if len(history) >= 60:
        returns = history['Close'].pct_change().tail(60)
        volatility_60d = returns.rolling(20).std() * np.sqrt(252) * 100
        cum_returns = ((1 + returns).cumprod() - 1) * 100
        
        fig.add_trace(
            go.Scatter(x=volatility_60d, y=cum_returns,
                      mode='markers', name='Risk-Return Profile',
                      marker=dict(size=8, color=cum_returns, colorscale='RdYlGn')),
            row=4, col=2
        )
    
    # Update layout for better appearance
    fig.update_layout(
        height=1200,
        title=f"📊 Comprehensive Analysis: {stock_data['symbol'].replace('.JK', '')}",
        showlegend=True,
        template='plotly_white',
        font=dict(family="Inter, sans-serif"),
        title_font_size=24,
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    # Update polar subplot
    fig.update_polars(
        radialaxis=dict(visible=True, range=[0, 100]),
        bgcolor='rgba(245,245,245,0.8)'
    )
    
    return fig

def create_portfolio_heatmap(analysis_results):
    """Create portfolio performance heatmap"""
    
    # Prepare data for heatmap
    symbols = []
    categories = []
    scores = []
    margins = []
    returns = []
    
    for result in analysis_results:
        symbols.append(result['symbol'].replace('.JK', ''))
        categories.append(result.get('category', 'Unknown'))
        scores.append(result['buffett_scores']['total_score'])
        margins.append(result['avg_margin'])
        returns.append(result.get('ml_return', 0))
    
    # Create heatmap data
    heatmap_data = pd.DataFrame({
        'Symbol': symbols,
        'Category': categories,
        'Buffett Score': scores,
        'Margin of Safety': margins,
        'ML Return Prediction': returns
    })
    
    # Create subplots for multiple heatmaps
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Buffett Score Heatmap', 'Margin of Safety Heatmap',
            'ML Return Prediction', 'Overall Investment Score'
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Calculate overall investment score
    heatmap_data['Overall Score'] = (
        heatmap_data['Buffett Score'] * 0.4 +
        heatmap_data['Margin of Safety'].clip(lower=0) * 0.3 +
        heatmap_data['ML Return Prediction'].clip(lower=0) * 3.0  # Scale ML returns
    )
    
    # Individual heatmaps
    for i, (col, title) in enumerate([
        ('Buffett Score', 'Buffett Score'),
        ('Margin of Safety', 'Margin of Safety (%)'),
        ('ML Return Prediction', 'Predicted Return (%)'),
        ('Overall Score', 'Investment Score')
    ]):
        row = (i // 2) + 1
        col_pos = (i % 2) + 1
        
        fig.add_trace(
            go.Heatmap(
                z=[heatmap_data[col].values],
                x=heatmap_data['Symbol'],
                y=['Score'],
                colorscale='RdYlGn',
                text=[heatmap_data[col].round(1).values],
                texttemplate="%{text}",
                textfont={"size": 12},
                showscale=True if i == 0 else False
            ), row=row, col=col_pos
        )
    
    fig.update_layout(
        title="Portfolio Performance Matrix",
        height=600,
        font=dict(family="Inter, sans-serif")
    )
    
    return fig

def create_advanced_comparison_dashboard(all_stock_data, analyzer):
    """Create advanced comparison dashboard"""
    
    comparison_data = []
    
    for symbol, stock_data in all_stock_data.items():
        if stock_data is None:
            continue
            
        metrics = analyzer.calculate_comprehensive_metrics(stock_data)
        scores = analyzer.enhanced_buffett_score(metrics)
        intrinsic_values = analyzer.multi_method_intrinsic_value(stock_data)
        current_price = stock_data['history']['Close'].iloc[-1]
        
        # Calculate average intrinsic value
        valid_values = [v for v in intrinsic_values.values() if v and v > 0]
        avg_intrinsic = np.mean(valid_values) if valid_values else current_price
        
        margin_safety = (avg_intrinsic - current_price) / avg_intrinsic * 100 if avg_intrinsic > 0 else 0
        
        # Get stock tier and market cap info
        symbol_info = None
        for category, stocks in INDONESIAN_STOCKS.items():
            if symbol in stocks:
                symbol_info = stocks[symbol]
                break
        
        tier = symbol_info['tier'] if symbol_info else 'Unknown'
        market_cap_size = symbol_info['market_cap'] if symbol_info else 'Unknown'
        
        comparison_data.append({
            'Symbol': symbol.replace('.JK', ''),
            'Company': symbol_info['name'] if symbol_info else symbol,
            'Tier': tier,
            'Market Cap': market_cap_size,
            'Current Price': current_price,
            'Buffett Score': scores['total_score'],
            'Value Score': scores['value_score'],
            'Quality Score': scores['quality_score'],
            'Growth Score': scores['growth_score'],
            'Financial Score': scores['financial_score'],
            'P/E': metrics.get('pe_ratio', 0),
            'P/B': metrics.get('pb_ratio', 0),
            'ROE': metrics.get('roe', 0),
            'ROA': metrics.get('roa', 0),
            'Debt/Equity': metrics.get('debt_to_equity', 0),
            'Margin Safety': margin_safety,
            'Dividend Yield': metrics.get('dividend_yield', 0),
            'Free FCF Yield': metrics.get('fcf_yield', 0),
            'Beta': metrics.get('beta', 1),
            'Volatility': metrics.get('volatility', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    
    if df.empty:
        return None
    
    # Create multiple comparison charts
    fig_comparison = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Quality vs Value Matrix', 'Risk-Return Profile',
            'Dividend Yield vs Growth', 'Market Cap vs Performance'
        )
    )
    
    # Quality vs Value scatter
    fig_comparison.add_trace(
        go.Scatter(
            x=df['Value Score'],
            y=df['Quality Score'],
            mode='markers+text',
            text=df['Symbol'],
            textposition='top center',
            marker=dict(
                size=df['Buffett Score']/2,
                color=df['Margin Safety'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Margin of Safety (%)")
            ),
            name='Stocks',
            hovertemplate='<b>%{text}</b><br>' +
                         'Value Score: %{x}<br>' +
                         'Quality Score: %{y}<br>' +
                         'Margin Safety: %{marker.color:.1f}%<extra></extra>'
        ), row=1, col=1
    )
    
    # Risk-Return Profile
    fig_comparison.add_trace(
        go.Scatter(
            x=df['Volatility'],
            y=df['ROE'],
            mode='markers+text',
            text=df['Symbol'],
            textposition='top center',
            marker=dict(
                size=df['Market Cap'].map({'Large': 20, 'Medium': 15, 'Small': 10}),
                color=df['Beta'],
                colorscale='RdYlBu_r',
                showscale=False
            ),
            name='Risk-Return',
            hovertemplate='<b>%{text}</b><br>' +
                         'Volatility: %{x:.2f}<br>' +
                         'ROE: %{y:.1f}%<br>' +
                         'Beta: %{marker.color:.2f}<extra></extra>'
        ), row=1, col=2
    )
    
    # Dividend vs Growth
    fig_comparison.add_trace(
        go.Scatter(
            x=df['Dividend Yield'],
            y=df['Growth Score'],
            mode='markers+text',
            text=df['Symbol'],
            textposition='top center',
            marker=dict(
                size=df['Financial Score']/3,
                color=df['Tier'].map({
                    'Blue Chip': 'blue', 'Growth': 'green', 'BUMN': 'orange',
                    'Dividend': 'purple', 'Commodity': 'brown', 'Recovery': 'red',
                    'Speculative': 'gray'
                }),
                opacity=0.7
            ),
            name='Dividend-Growth',
            hovertemplate='<b>%{text}</b><br>' +
                         'Dividend Yield: %{x:.1f}%<br>' +
                         'Growth Score: %{y}<br>' +
                         'Tier: %{marker.color}<extra></extra>'
        ), row=2, col=1
    )
    
    # Market Cap vs Performance
    market_cap_numeric = df['Market Cap'].map({'Large': 3, 'Medium': 2, 'Small': 1})
    
    fig_comparison.add_trace(
        go.Scatter(
            x=market_cap_numeric,
            y=df['Buffett Score'],
            mode='markers+text',
            text=df['Symbol'],
            textposition='top center',
            marker=dict(
                size=df['Margin Safety'].clip(lower=0)/2 + 5,
                color=df['Overall Score'] if 'Overall Score' in df.columns else df['Buffett Score'],
                colorscale='Viridis',
                showscale=False
            ),
            name='Cap-Performance',
            hovertemplate='<b>%{text}</b><br>' +
                         'Market Cap: %{x}<br>' +
                         'Buffett Score: %{y}<br>' +
                         'Margin Safety: %{marker.size:.1f}%<extra></extra>'
        ), row=2, col=2
    )
    
    # Update axes
    fig_comparison.update_xaxes(title_text="Value Score", row=1, col=1)
    fig_comparison.update_yaxes(title_text="Quality Score", row=1, col=1)
    fig_comparison.update_xaxes(title_text="Volatility", row=1, col=2)
    fig_comparison.update_yaxes(title_text="ROE (%)", row=1, col=2)
    fig_comparison.update_xaxes(title_text="Dividend Yield (%)", row=2, col=1)
    fig_comparison.update_yaxes(title_text="Growth Score", row=2, col=1)
    fig_comparison.update_xaxes(title_text="Market Cap Size", tickvals=[1,2,3], 
                               ticktext=['Small','Medium','Large'], row=2, col=2)
    fig_comparison.update_yaxes(title_text="Buffett Score", row=2, col=2)
    
    fig_comparison.update_layout(
        height=800,
        title="Multi-Dimensional Stock Comparison Matrix",
        showlegend=False,
        template='plotly_white'
    )
    
    return fig_comparison

def create_risk_analysis_chart(metrics, stock_info):
    """Create comprehensive risk analysis visualization"""
    
    # Risk categories and scores
    risk_categories = {
        'Financial Risk': min(metrics.get('debt_to_equity', 0) / 50 * 100, 100),
        'Liquidity Risk': max(0, 100 - metrics.get('current_ratio', 0) * 40),
        'Valuation Risk': min(metrics.get('pe_ratio', 0) / 30 * 100, 100),
        'Market Risk': min(metrics.get('beta', 1) * 50, 100),
        'Volatility Risk': min(metrics.get('max_drawdown', 0), 100),
        'Operational Risk': max(0, 100 - metrics.get('roe', 0) * 4)
    }
    
    # Create risk radar chart
    fig_risk = go.Figure()
    
    fig_risk.add_trace(go.Scatterpolar(
        r=list(risk_categories.values()),
        theta=list(risk_categories.keys()),
        fill='toself',
        name='Risk Profile',
        line_color='red',
        fillcolor='rgba(255,0,0,0.25)'
    ))
    
    # Add benchmark (low risk profile)
    benchmark_risk = [20] * len(risk_categories)  # 20% risk across all categories
    
    fig_risk.add_trace(go.Scatterpolar(
        r=benchmark_risk,
        theta=list(risk_categories.keys()),
        fill='toself',
        name='Low Risk Benchmark',
        line_color='green',
        fillcolor='rgba(0,255,0,0.1)'
    ))
    
    fig_risk.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickmode='linear',
                tick0=0,
                dtick=20
            )
        ),
        title="Comprehensive Risk Analysis",
        height=500
    )
    
    return fig_risk

def main():
    st.markdown('<h1 class="main-header">Warren Buffett Stock Analyzer Pro</h1>', unsafe_allow_html=True)
    
    # Enhanced info banner with live status
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.success("Real-time market data powered by Yahoo Finance API")
    with col2:
        st.info(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    with col3:
        if st.button("Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Enhanced sidebar with better organization
    st.sidebar.markdown("## Configuration Panel")
    
    # Category selection with enhanced info
    st.sidebar.markdown("### Stock Selection")
    
    # Multi-category selection
    available_categories = list(INDONESIAN_STOCKS.keys())
    selected_categories = st.sidebar.multiselect(
        "Select Categories:",
        available_categories,
        default=[available_categories[0]],
        help="Choose one or more sectors for cross-sector analysis"
    )
    
    if not selected_categories:
        st.sidebar.warning("Please select at least one category")
        return
    
    # Aggregate stocks from selected categories
    all_available_stocks = {}
    for category in selected_categories:
        for symbol, info in INDONESIAN_STOCKS[category].items():
            all_available_stocks[symbol] = {**info, 'category': category}
    
    # Stock selection with filtering options
    st.sidebar.markdown("### Advanced Filtering")
    
    # Filter by tier
    available_tiers = list(set([info['tier'] for info in all_available_stocks.values()]))
    selected_tiers = st.sidebar.multiselect(
        "Filter by Tier:",
        available_tiers,
        default=available_tiers,
        help="Filter stocks by quality tier"
    )
    
    # Filter by market cap
    available_caps = list(set([info['market_cap'] for info in all_available_stocks.values()]))
    selected_caps = st.sidebar.multiselect(
        "Filter by Market Cap:",
        available_caps,
        default=available_caps,
        help="Filter stocks by market capitalization"
    )
    
    # Apply filters
    filtered_stocks = {
        symbol: info for symbol, info in all_available_stocks.items()
        if info['tier'] in selected_tiers and info['market_cap'] in selected_caps
    }
    
    # Final stock selection
    stock_options = list(filtered_stocks.keys())
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks for Analysis:",
        stock_options,
        default=stock_options[:min(8, len(stock_options))],
        help="Select stocks for detailed analysis (max 15 for optimal performance)"
    )
    
    if len(selected_stocks) > 15:
        st.sidebar.warning("Maximum 15 stocks recommended for optimal performance")
        selected_stocks = selected_stocks[:15]
    
    # Enhanced Warren Buffett parameters
    st.sidebar.markdown("### Warren Buffett Criteria")
    
    with st.sidebar.expander("Value Metrics", expanded=True):
        max_pe = st.slider("Maximum P/E Ratio", 5, 50, 20, help="Buffett prefers P/E < 15")
        max_pb = st.slider("Maximum P/B Ratio", 0.5, 10.0, 3.0, help="Buffett likes P/B < 1.5")
        min_margin = st.slider("Minimum Margin of Safety (%)", 0, 50, 25, help="Buffett's safety margin")
    
    with st.sidebar.expander("Quality Metrics"):
        min_roe = st.slider("Minimum ROE (%)", 5, 30, 15, help="Buffett seeks ROE > 15%")
        min_roa = st.slider("Minimum ROA (%)", 2, 20, 8, help="Asset efficiency indicator")
        max_debt = st.slider("Maximum Debt/Equity (%)", 10, 100, 40, help="Buffett avoids high debt")
    
    with st.sidebar.expander("Growth & Profitability"):
        min_profit_margin = st.slider("Minimum Profit Margin (%)", 0, 30, 10)
        min_revenue_growth = st.slider("Minimum Revenue Growth (%)", -10, 30, 5)
        min_consistency = st.slider("Minimum Consistency Score", 0, 100, 60)
    
    # Analysis configuration
    st.sidebar.markdown("### Analysis Configuration")
    
    analysis_mode = st.sidebar.radio(
        "Analysis Mode:",
        ["Conservative (Pure Buffett)", "Balanced", "Growth-Enhanced"],
        index=1,
        help="Choose analysis approach"
    )
    
    use_ml = st.sidebar.checkbox("Enable ML Predictions", value=True)
    use_technical = st.sidebar.checkbox("Include Technical Analysis", value=True)
    include_risk_analysis = st.sidebar.checkbox("Comprehensive Risk Analysis", value=True)
    
    # Prediction timeframe
    if use_ml:
        prediction_horizon = st.sidebar.selectbox(
            "ML Prediction Horizon:",
            ["1m", "3m", "6m"],
            index=1,
            help="Choose prediction timeframe"
        )
    
    # Main analysis
    if not selected_stocks:
        st.warning("Please select at least one stock for analysis")
        return
    
    # Load data with enhanced progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Loading stock data...")
    all_stock_data = load_enhanced_stock_data(selected_stocks, progress_bar)
    
    if not all_stock_data:
        st.error("Failed to load stock data. Please check your internet connection.")
        return
    
    progress_bar.progress(100)
    status_text.text(f"Successfully loaded {len(all_stock_data)} stocks")
    
    # Initialize enhanced analyzer
    analyzer = EnhancedWarrenBuffettAnalyzer()
    
    # Train ML models if enabled
    ml_results = None
    if use_ml:
        with st.spinner('Training advanced ML ensemble...'):
            ml_results = analyzer.train_advanced_ensemble_model(all_stock_data)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dashboard", "Screening", "ML Analysis", "Individual Analysis", "Risk Analysis", "Export & Reports"
    ])
    
    with tab1:
        st.markdown("## Executive Dashboard")
        
        # Key metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate portfolio metrics
        total_analyzed = len(all_stock_data)
        high_quality = 0
        undervalued = 0
        ml_positive = 0
        
        for symbol, stock_data in all_stock_data.items():
            metrics = analyzer.calculate_comprehensive_metrics(stock_data)
            scores = analyzer.enhanced_buffett_score(metrics)
            intrinsic_values = analyzer.multi_method_intrinsic_value(stock_data)
            current_price = stock_data['history']['Close'].iloc[-1]
            
            if scores['total_score'] >= 70:
                high_quality += 1
            
            # Calculate margin of safety
            valid_values = [v for v in intrinsic_values.values() if v and v > 0]
            if valid_values:
                avg_intrinsic = np.mean(valid_values)
                margin = (avg_intrinsic - current_price) / avg_intrinsic * 100
                if margin > min_margin:
                    undervalued += 1
            
            # ML prediction
            if use_ml and analyzer.model:
                ml_pred = analyzer.predict_enhanced_returns(stock_data, prediction_horizon)
                if ml_pred and ml_pred['prediction'] > 0.05:
                    ml_positive += 1
        
        with col1:
            st.metric("Total Analyzed", total_analyzed, help="Number of stocks analyzed")
        with col2:
            st.metric("High Quality", high_quality, f"{high_quality/total_analyzed*100:.1f}%")
        with col3:
            st.metric("Undervalued", undervalued, f"{undervalued/total_analyzed*100:.1f}%")
        with col4:
            if use_ml:
                st.metric("ML Positive", ml_positive, f"{ml_positive/total_analyzed*100:.1f}%")
        
        # Market overview heatmap
        st.markdown("### Market Overview Heatmap")
        
        # Prepare heatmap data
        heatmap_data = []
        for symbol, stock_data in all_stock_data.items():
            metrics = analyzer.calculate_comprehensive_metrics(stock_data)
            scores = analyzer.enhanced_buffett_score(metrics)
            
            stock_info = filtered_stocks.get(symbol, {})
            
            heatmap_data.append({
                'Symbol': symbol.replace('.JK', ''),
                'Category': stock_info.get('category', 'Unknown'),
                'Tier': stock_info.get('tier', 'Unknown'),
                'Buffett Score': scores['total_score'],
                'P/E': metrics.get('pe_ratio', 0),
                'ROE': metrics.get('roe', 0)
            })
        
        df_heatmap = pd.DataFrame(heatmap_data)
        
        # Create interactive heatmap
        fig_heatmap = px.treemap(
            df_heatmap,
            path=['Category', 'Tier', 'Symbol'],
            values='Buffett Score',
            color='ROE',
            color_continuous_scale='RdYlGn',
            title="Portfolio Composition by Quality"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Top performers summary
        st.markdown("### Top Performers")
        
        # Calculate and sort by composite score
        performance_data = []
        for symbol, stock_data in all_stock_data.items():
            metrics = analyzer.calculate_comprehensive_metrics(stock_data)
            scores = analyzer.enhanced_buffett_score(metrics)
            intrinsic_values = analyzer.multi_method_intrinsic_value(stock_data)
            current_price = stock_data['history']['Close'].iloc[-1]
            
            # Calculate composite opportunity score
            valid_values = [v for v in intrinsic_values.values() if v and v > 0]
            avg_margin = 0
            if valid_values:
                avg_intrinsic = np.mean(valid_values)
                avg_margin = (avg_intrinsic - current_price) / avg_intrinsic * 100
            
            ml_return = 0
            if use_ml and analyzer.model:
                ml_pred = analyzer.predict_enhanced_returns(stock_data, prediction_horizon)
                if ml_pred:
                    ml_return = ml_pred['prediction'] * 100
            
            composite_score = (
                scores['total_score'] * 0.4 +
                max(0, avg_margin) * 0.3 +
                max(0, ml_return * 2) * 0.3
            )
            
            performance_data.append({
                'Symbol': symbol.replace('.JK', ''),
                'Company': filtered_stocks.get(symbol, {}).get('name', symbol),
                'Composite Score': composite_score,
                'Buffett Score': scores['total_score'],
                'Margin Safety': avg_margin,
                'ML Return': ml_return,
                'Current Price': current_price
            })
        
        # Sort and display top 5
        top_performers = sorted(performance_data, key=lambda x: x['Composite Score'], reverse=True)[:5]
        
        for i, stock in enumerate(top_performers):
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
                
                with col1:
                    st.markdown(f"**#{i+1} {stock['Symbol']}**")
                    st.caption(stock['Company'][:40] + "..." if len(stock['Company']) > 40 else stock['Company'])
                
                with col2:
                    score_color = "🟢" if stock['Buffett Score'] >= 80 else "🟡" if stock['Buffett Score'] >= 60 else "🔴"
                    st.metric("Quality", f"{score_color} {stock['Buffett Score']}/100")
                
                with col3:
                    margin_color = "🟢" if stock['Margin Safety'] > 25 else "🟡" if stock['Margin Safety'] > 0 else "🔴"
                    st.metric("Value", f"{margin_color} {stock['Margin Safety']:.1f}%")
                
                with col4:
                    if use_ml:
                        ml_color = "🟢" if stock['ML Return'] > 8 else "🟡" if stock['ML Return'] > 0 else "🔴"
                        st.metric("ML Signal", f"{ml_color} {stock['ML Return']:.1f}%")
                
                with col5:
                    st.metric("Price", f"Rp {stock['Current Price']:,.0f}")
                
                st.markdown("---")
    
    with tab2:
        st.markdown("## Advanced Stock Screening")
        
        # Enhanced filtering interface
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sort_metric = st.selectbox("Sort by:", [
                "Composite Score", "Buffett Score", "Margin of Safety", 
                "ML Prediction", "Risk-Adjusted Return"
            ])
        
        with col2:
            filter_quality = st.selectbox("Quality Filter:", [
                "All", "High Quality (70+)", "Medium Quality (50-70)", "Low Quality (<50)"
            ])
        
        with col3:
            filter_value = st.selectbox("Value Filter:", [
                "All", "Deeply Undervalued (25%+)", "Undervalued (10%+)", "Fair Value", "Overvalued"
            ])
        
        with col4:
            show_only_buys = st.checkbox("Show Only Buy Recommendations")
        
        # Perform comprehensive analysis
        analysis_results = []
        
        for symbol, stock_data in all_stock_data.items():
            metrics = analyzer.calculate_comprehensive_metrics(stock_data)
            scores = analyzer.enhanced_buffett_score(metrics)
            intrinsic_values = analyzer.multi_method_intrinsic_value(stock_data)
            current_price = stock_data['history']['Close'].iloc[-1]
            
            # Calculate margins
            valid_values = [v for v in intrinsic_values.values() if v and v > 0]
            avg_margin = 0
            if valid_values:
                avg_intrinsic = np.mean(valid_values)
                avg_margin = (avg_intrinsic - current_price) / avg_intrinsic * 100
            
            # ML prediction
            ml_pred_data = None
            ml_return = 0
            if use_ml and analyzer.model:
                ml_pred_data = analyzer.predict_enhanced_returns(stock_data, prediction_horizon)
                if ml_pred_data:
                    ml_return = ml_pred_data['prediction'] * 100
            
            # Get recommendation
            stock_info = filtered_stocks.get(symbol, {})
            recommendation, reason, rec_score = analyzer.enhanced_investment_recommendation(
                metrics, intrinsic_values, current_price, ml_pred_data, stock_info
            )
            
            # Risk assessment
            risk_score, risk_factors = analyzer.comprehensive_risk_assessment(metrics, stock_info)
            
            analysis_results.append({
                'symbol': symbol,
                'company': stock_info.get('name', symbol),
                'category': stock_info.get('category', 'Unknown'),
                'tier': stock_info.get('tier', 'Unknown'),
                'current_price': current_price,
                'buffett_scores': scores,
                'metrics': metrics,
                'avg_margin': avg_margin,
                'ml_return': ml_return,
                'ml_data': ml_pred_data,
                'recommendation': recommendation,
                'reason': reason,
                'rec_score': rec_score,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'composite_score': (scores['total_score'] * 0.4 + max(0, avg_margin) * 0.3 + 
                                  max(0, ml_return * 2) * 0.3)
            })
        
        # Apply filters
        filtered_results = analysis_results.copy()
        
        # Quality filter
        if filter_quality == "High Quality (70+)":
            filtered_results = [r for r in filtered_results if r['buffett_scores']['total_score'] >= 70]
        elif filter_quality == "Medium Quality (50-70)":
            filtered_results = [r for r in filtered_results if 50 <= r['buffett_scores']['total_score'] < 70]
        elif filter_quality == "Low Quality (<50)":
            filtered_results = [r for r in filtered_results if r['buffett_scores']['total_score'] < 50]
        
        # Value filter
        if filter_value == "Deeply Undervalued (25%+)":
            filtered_results = [r for r in filtered_results if r['avg_margin'] > 25]
        elif filter_value == "Undervalued (10%+)":
            filtered_results = [r for r in filtered_results if r['avg_margin'] > 10]
        elif filter_value == "Fair Value":
            filtered_results = [r for r in filtered_results if -10 <= r['avg_margin'] <= 10]
        elif filter_value == "Overvalued":
            filtered_results = [r for r in filtered_results if r['avg_margin'] < -10]
        
        # Buy recommendations filter
        if show_only_buys:
            filtered_results = [r for r in filtered_results if "BUY" in r['recommendation']]
        
        # Sort results
        sort_keys = {
            "Composite Score": lambda x: x['composite_score'],
            "Buffett Score": lambda x: x['buffett_scores']['total_score'],
            "Margin of Safety": lambda x: x['avg_margin'],
            "ML Prediction": lambda x: x['ml_return'],
            "Risk-Adjusted Return": lambda x: x['ml_return'] - x['risk_score']/10
        }
        
        filtered_results.sort(key=sort_keys[sort_metric], reverse=True)
        
        # Display results with enhanced cards
        st.markdown(f"### Screening Results ({len(filtered_results)} stocks)")
        
        for result in filtered_results:
            with st.container():
                # Determine card styling based on recommendation
                card_class = "strong-buy" if "STRONG BUY" in result['recommendation'] else \
                           "buy" if "BUY" in result['recommendation'] else \
                           "hold" if "HOLD" in result['recommendation'] else \
                           "sell" if "SELL" in result['recommendation'] else "metric-card"
                
                st.markdown(f'<div class="metric-card {card_class}">', unsafe_allow_html=True)
                
                # Header row
                col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
                
                with col1:
                    st.markdown(f"### {result['symbol']} - {result['tier']}")
                    st.caption(f"{result['company']} | {result['category']}")
                    st.metric("Current Price", f"Rp {result['current_price']:,.0f}")
                
                with col2:
                    st.markdown("**Quality Scores**")
                    st.metric("Total", f"{result['buffett_scores']['total_score']}/100")
                    st.metric("Value", f"{result['buffett_scores']['value_score']}/100")
                    st.metric("Quality", f"{result['buffett_scores']['quality_score']}/100")
                
                with col3:
                    st.markdown("**Investment Metrics**")
                    st.metric("Margin Safety", f"{result['avg_margin']:.1f}%")
                    if use_ml:
                        confidence = "High" if result['ml_data'] and result['ml_data']['uncertainty'] < 0.05 else "Medium"
                        st.metric("ML Return", f"{result['ml_return']:.1f}%", confidence)
                    st.metric("Risk Score", f"{result['risk_score']}/100")
                
                with col4:
                    st.markdown("**Recommendation**")
                    rec_color = "success" if "BUY" in result['recommendation'] else \
                              "warning" if "HOLD" in result['recommendation'] else \
                              "error" if "SELL" in result['recommendation'] else "info"
                    
                    getattr(st, rec_color)(result['recommendation'])
                    st.info(result['reason'])
                    
                    # Show key risk factors
                    if result['risk_factors']:
                        with st.expander("Risk Factors"):
                            for factor in result['risk_factors'][:3]:
                                st.caption(f"• {factor}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
        
        # Advanced comparison charts
        if len(filtered_results) > 1:
            st.markdown("### Portfolio Comparison Matrix")
            comparison_fig = create_advanced_comparison_dashboard(all_stock_data, analyzer)
            if comparison_fig:
                st.plotly_chart(comparison_fig, use_container_width=True)
    
    with tab3:
        if use_ml and ml_results:
            st.markdown("## Machine Learning Analysis")
            
            # Model performance overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_r2 = np.mean([scores['r2'] for scores in ml_results['cv_scores'].values()])
                st.metric("Avg Model Accuracy", f"{avg_r2:.3f}", help="Cross-validation R² score")
            
            with col2:
                st.metric("Training Samples", f"{ml_results['n_samples']}")
            
            with col3:
                st.metric("Selected Features", f"{ml_results['n_features_selected']}")
            
            with col4:
                best_timeframe = max(ml_results['cv_scores'].items(), key=lambda x: np.mean(list(x[1].values())))
                st.metric("Best Timeframe", best_timeframe[0])
            
            # Model performance comparison
            st.markdown("### Model Performance by Timeframe")
            
            perf_data = []
            for timeframe, scores in ml_results['cv_scores'].items():
                for model, score in scores.items():
                    perf_data.append({
                        'Timeframe': timeframe,
                        'Model': model,
                        'R² Score': score
                    })
            
            df_perf = pd.DataFrame(perf_data)
            
            fig_perf = px.bar(
                df_perf,
                x='Timeframe',
                y='R² Score',
                color='Model',
                barmode='group',
                title="Model Performance Comparison"
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Feature importance analysis
            if not ml_results['feature_importance'].empty:
                st.markdown("### Feature Importance Analysis")
                
                fig_importance = px.bar(
                    ml_results['feature_importance'].head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 15 Most Important Features",
                    color='importance',
                    color_continuous_scale='viridis'
                )
                fig_importance.update_layout(height=600)
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Individual predictions with confidence intervals
            st.markdown("### Detailed ML Predictions")
            
            pred_data = []
            for result in analysis_results:
                if result['ml_data']:
                    pred_data.append({
                        'Symbol': result['symbol'].replace('.JK', ''),
                        'Prediction': f"{result['ml_return']:.2f}%",
                        'Confidence Lower': f"{result['ml_data']['confidence_interval']['lower']*100:.2f}%",
                        'Confidence Upper': f"{result['ml_data']['confidence_interval']['upper']*100:.2f}%",
                        'Uncertainty': f"{result['ml_data']['uncertainty']*100:.3f}%",
                        'Recommendation': result['recommendation']
                    })
            
            if pred_data:
                df_pred = pd.DataFrame(pred_data)
                st.dataframe(
                    df_pred,
                    use_container_width=True,
                    column_config={
                        'Symbol': st.column_config.TextColumn('Stock', width="small"),
                        'Prediction': st.column_config.TextColumn('Expected Return'),
                        'Confidence Lower': st.column_config.TextColumn('Lower Bound'),
                        'Confidence Upper': st.column_config.TextColumn('Upper Bound'),
                        'Uncertainty': st.column_config.TextColumn('Uncertainty'),
                        'Recommendation': st.column_config.TextColumn('Action')
                    }
                )
                
                # Prediction distribution analysis
                predictions = [float(p['Prediction'].replace('%', '')) for p in pred_data]
                
                fig_dist = px.histogram(
                    x=predictions,
                    nbins=15,
                    title=f"Distribution of {prediction_horizon.upper()} Return Predictions",
                    labels={'x': 'Predicted Return (%)', 'y': 'Number of Stocks'}
                )
                fig_dist.add_vline(x=np.mean(predictions), line_dash="dash", line_color="red", 
                                 annotation_text=f"Mean: {np.mean(predictions):.1f}%")
                fig_dist.add_vline(x=np.median(predictions), line_dash="dash", line_color="green",
                                 annotation_text=f"Median: {np.median(predictions):.1f}%")
                st.plotly_chart(fig_dist, use_container_width=True)
        
        else:
            st.warning("Machine Learning analysis is disabled. Enable it in the sidebar to see predictions.")
    
    with tab4:
        st.markdown("## Individual Stock Deep Dive")
        
        # Stock selection for detailed analysis
        selected_stock = st.selectbox(
            "Select stock for detailed analysis:",
            options=list(all_stock_data.keys()),
            format_func=lambda x: f"{x.replace('.JK', '')} - {filtered_stocks.get(x, {}).get('name', x)}"
        )
        
        if selected_stock and selected_stock in all_stock_data:
            stock_data = all_stock_data[selected_stock]
            
            if stock_data is None:
                st.error(f"No data available for {selected_stock}")
                return
            
            # Calculate comprehensive analysis
            metrics = analyzer.calculate_comprehensive_metrics(stock_data)
            scores = analyzer.enhanced_buffett_score(metrics)
            intrinsic_values = analyzer.multi_method_intrinsic_value(stock_data)
            current_price = stock_data['history']['Close'].iloc[-1]
            
            # ML prediction for this stock
            ml_pred = None
            if use_ml and analyzer.model:
                ml_pred = analyzer.predict_enhanced_returns(stock_data, prediction_horizon)
            
            # Get recommendation
            stock_info = filtered_stocks.get(selected_stock, {})
            recommendation, reason, rec_score = analyzer.enhanced_investment_recommendation(
                metrics, intrinsic_values, current_price, ml_pred, stock_info
            )
            
            # Header section with key info
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
            
            with col1:
                st.markdown(f"### {selected_stock.replace('.JK', '')}")
                st.caption(stock_info.get('name', selected_stock))
                st.caption(f"{stock_info.get('category', 'Unknown')} • {stock_info.get('tier', 'Unknown')} • {stock_info.get('market_cap', 'Unknown')} Cap")
            
            with col2:
                st.metric("Current Price", f"Rp {current_price:,.0f}")
                
            with col3:
                st.metric("Buffett Score", f"{scores['total_score']}/100")
                
            with col4:
                # Calculate average margin
                valid_values = [v for v in intrinsic_values.values() if v and v > 0]
                avg_margin = np.mean([(v - current_price) / v * 100 for v in valid_values]) if valid_values else 0
                margin_delta = "🟢" if avg_margin > 25 else "🟡" if avg_margin > 0 else "🔴"
                st.metric("Margin of Safety", f"{avg_margin:.1f}%", delta=margin_delta)
                
            with col5:
                rec_color = "success" if "BUY" in recommendation else "warning" if "HOLD" in recommendation else "error"
                st.metric("Recommendation", recommendation.split()[1] if len(recommendation.split()) > 1 else recommendation)
            
            st.markdown("---")
            
            # Detailed scores breakdown
            st.markdown("#### Warren Buffett Score Breakdown")
            score_col1, score_col2, score_col3, score_col4, score_col5 = st.columns(5)
            
            with score_col1:
                st.metric("Value Score", f"{scores['value_score']}/100", 
                         help="P/E, P/B, PEG ratios")
            with score_col2:
                st.metric("Quality Score", f"{scores['quality_score']}/100", 
                         help="ROE, ROA, ROIC, margins")
            with score_col3:
                st.metric("Growth Score", f"{scores['growth_score']}/100", 
                         help="Revenue/earnings growth, consistency")
            with score_col4:
                st.metric("Financial Score", f"{scores['financial_score']}/100", 
                         help="Debt levels, liquidity, coverage")
            with score_col5:
                st.metric("Management Score", f"{scores['management_score']}/100", 
                         help="Capital allocation efficiency")
            
            # Intrinsic value analysis
            st.markdown("#### Intrinsic Value Analysis")
            
            if intrinsic_values:
                iv_col1, iv_col2, iv_col3 = st.columns(3)
                
                # Display different valuation methods
                methods_display = {
                    'dcf_moderate': 'DCF (Moderate)',
                    'dcf_conservative': 'DCF (Conservative)',
                    'graham_enhanced': 'Graham Number',
                    'ddm': 'Dividend Model',
                    'relative_valuation': 'Relative Valuation',
                    'weighted_average': 'Weighted Average'
                }
                
                valid_methods = {k: v for k, v in intrinsic_values.items() if v and v > 0}
                
                with iv_col1:
                    st.markdown("**Valuation Methods**")
                    for method, value in valid_methods.items():
                        if method in methods_display:
                            margin = (value - current_price) / value * 100 if value > 0 else 0
                            color = "🟢" if margin > 25 else "🟡" if margin > 0 else "🔴"
                            st.metric(methods_display[method], f"Rp {value:,.0f}", 
                                    delta=f"{color} {margin:.1f}%")
                
                with iv_col2:
                    # Valuation range visualization
                    if len(valid_methods) > 1:
                        values = list(valid_methods.values())
                        fig_range = go.Figure()
                        
                        fig_range.add_trace(go.Box(
                            y=values,
                            name="Intrinsic Value Range",
                            boxpoints='all',
                            jitter=0.3,
                            pointpos=-1.8
                        ))
                        
                        fig_range.add_hline(y=current_price, line_dash="dash", line_color="red",
                                          annotation_text=f"Current: Rp {current_price:,.0f}")
                        
                        fig_range.update_layout(
                            title="Valuation Range",
                            yaxis_title="Price (Rp)",
                            height=300,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_range, use_container_width=True)
                
                with iv_col3:
                    st.markdown("**Value Interpretation**")
                    if avg_margin > 30:
                        st.success("Deeply Undervalued - Strong Buy Signal")
                    elif avg_margin > 15:
                        st.info("Undervalued - Buy Signal")  
                    elif avg_margin > -10:
                        st.warning("Fair Value - Hold")
                    else:
                        st.error("Overvalued - Sell Signal")
                    
                    st.markdown("**Key Insights**")
                    st.write(f"• Average margin: {avg_margin:.1f}%")
                    st.write(f"• Consensus methods: {len(valid_methods)}")
                    st.write(f"• Confidence: {'High' if len(valid_methods) >= 3 else 'Medium'}")
            
            # Advanced technical and fundamental charts
            st.markdown("#### Comprehensive Charts")
            
            advanced_fig = create_enhanced_visualizations(stock_data, metrics, analyzer)
            st.plotly_chart(advanced_fig, use_container_width=True)
            
            # Key financial metrics table
            st.markdown("#### Key Financial Metrics")
            
            metrics_data = {
                'Valuation': {
                    'P/E Ratio': f"{metrics.get('pe_ratio', 0):.1f}",
                    'P/B Ratio': f"{metrics.get('pb_ratio', 0):.1f}", 
                    'PEG Ratio': f"{metrics.get('peg_ratio', 0):.1f}",
                    'P/S Ratio': f"{metrics.get('price_to_sales', 0):.1f}",
                    'EV/EBITDA': f"{metrics.get('ev_to_ebitda', 0):.1f}"
                },
                'Profitability': {
                    'ROE': f"{metrics.get('roe', 0):.1f}%",
                    'ROA': f"{metrics.get('roa', 0):.1f}%",
                    'ROIC': f"{metrics.get('roic', 0):.1f}%",
                    'Profit Margin': f"{metrics.get('profit_margin', 0):.1f}%",
                    'Operating Margin': f"{metrics.get('operating_margin', 0):.1f}%"
                },
                'Financial Health': {
                    'Current Ratio': f"{metrics.get('current_ratio', 0):.1f}",
                    'Quick Ratio': f"{metrics.get('quick_ratio', 0):.1f}",
                    'Debt/Equity': f"{metrics.get('debt_to_equity', 0):.1f}%",
                    'Interest Coverage': f"{metrics.get('interest_coverage', 0):.1f}x",
                    'Free FCF Yield': f"{metrics.get('fcf_yield', 0):.1f}%"
                },
                'Growth & Returns': {
                    'Revenue Growth': f"{metrics.get('revenue_growth', 0):.1f}%",
                    'Earnings Growth': f"{metrics.get('earnings_growth', 0):.1f}%",
                    'Dividend Yield': f"{metrics.get('dividend_yield', 0):.1f}%",
                    'Beta': f"{metrics.get('beta', 1):.2f}",
                    'Volatility': f"{metrics.get('volatility', 0):.1f}%"
                }
            }
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            for i, (category, category_metrics) in enumerate(metrics_data.items()):
                with [metric_col1, metric_col2, metric_col3, metric_col4][i]:
                    st.markdown(f"**{category}**")
                    for metric_name, value in category_metrics.items():
                        st.caption(f"{metric_name}: {value}")
            
            # ML Analysis section (if enabled)
            if use_ml and ml_pred:
                st.markdown("#### Machine Learning Analysis")
                
                ml_col1, ml_col2, ml_col3 = st.columns(3)
                
                with ml_col1:
                    prediction_pct = ml_pred['prediction'] * 100
                    color = "success" if prediction_pct > 5 else "warning" if prediction_pct > 0 else "error"
                    getattr(st, color)(f"**{prediction_horizon.upper()} Prediction: {prediction_pct:.2f}%**")
                    
                    confidence = 100 - (ml_pred['uncertainty'] * 1000)
                    st.metric("Confidence Level", f"{max(0, min(confidence, 100)):.0f}%")
                
                with ml_col2:
                    st.markdown("**Confidence Interval**")
                    lower = ml_pred['confidence_interval']['lower'] * 100
                    upper = ml_pred['confidence_interval']['upper'] * 100
                    st.write(f"Lower: {lower:.2f}%")
                    st.write(f"Upper: {upper:.2f}%")
                    st.write(f"Range: ±{(upper-lower)/2:.1f}%")
                
                with ml_col3:
                    st.markdown("**Individual Model Predictions**")
                    for model, pred in ml_pred['individual_predictions'].items():
                        st.caption(f"{model.upper()}: {pred*100:.2f}%")
            
            # Investment summary and recommendation
            st.markdown("#### Investment Summary")
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown("**Strengths**")
                strengths = []
                
                if scores['total_score'] >= 80:
                    strengths.append("Exceptional fundamental quality")
                elif scores['total_score'] >= 65:
                    strengths.append("Strong fundamental quality")
                
                if avg_margin > 25:
                    strengths.append("High margin of safety")
                elif avg_margin > 10:
                    strengths.append("Adequate margin of safety")
                
                if metrics.get('roe', 0) > 20:
                    strengths.append("Excellent return on equity")
                
                if metrics.get('debt_to_equity', 100) < 30:
                    strengths.append("Conservative debt levels")
                
                if use_ml and ml_pred and ml_pred['prediction'] > 0.08:
                    strengths.append("Strong ML prediction signal")
                
                for strength in strengths[:5]:
                    st.success(f"✅ {strength}")
            
            with summary_col2:
                st.markdown("**Areas of Concern**")
                concerns = []
                
                if scores['total_score'] < 50:
                    concerns.append("Weak fundamental metrics")
                
                if avg_margin < -15:
                    concerns.append("Significantly overvalued")
                
                if metrics.get('debt_to_equity', 0) > 60:
                    concerns.append("High debt burden")
                
                if metrics.get('current_ratio', 2) < 1.0:
                    concerns.append("Liquidity concerns")
                
                if metrics.get('pe_ratio', 0) > 30:
                    concerns.append("High valuation multiples")
                
                if use_ml and ml_pred and ml_pred['prediction'] < -0.05:
                    concerns.append("Negative ML prediction")
                
                for concern in concerns[:5]:
                    st.error(f"⚠️ {concern}")
                
                if not concerns:
                    st.info("No major concerns identified")
            
            # Final recommendation box
            rec_color_map = {
                "STRONG BUY": "success",
                "BUY": "info", 
                "HOLD": "warning",
                "SELL": "error",
                "WATCH": "secondary"
            }
            
            rec_type = recommendation.split()[1] if len(recommendation.split()) > 1 else recommendation
            rec_color = rec_color_map.get(rec_type, "info")
            
            getattr(st, rec_color)(f"**Final Recommendation: {recommendation}**")
            st.info(f"Reasoning: {reason}")
            st.caption(f"Recommendation confidence: {rec_score:.0f}/100")
    
    with tab5:
        if include_risk_analysis:
            st.markdown("## Comprehensive Risk Analysis")
            
            # Portfolio-level risk metrics
            st.markdown("### Portfolio Risk Overview")
            
            portfolio_risks = []
            high_risk_stocks = []
            
            for symbol, stock_data in all_stock_data.items():
                metrics = analyzer.calculate_comprehensive_metrics(stock_data)
                stock_info = filtered_stocks.get(symbol, {})
                risk_score, risk_factors = analyzer.comprehensive_risk_assessment(metrics, stock_info)
                
                portfolio_risks.append(risk_score)
                if risk_score > 70:
                    high_risk_stocks.append({
                        'symbol': symbol.replace('.JK', ''),
                        'risk_score': risk_score,
                        'factors': risk_factors[:3]
                    })
            
            # Portfolio risk metrics
            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
            
            with risk_col1:
                avg_risk = np.mean(portfolio_risks)
                risk_color = "error" if avg_risk > 60 else "warning" if avg_risk > 40 else "success"
                getattr(st, risk_color)(f"Portfolio Risk Score: {avg_risk:.0f}/100")
            
            with risk_col2:
                high_risk_pct = len(high_risk_stocks) / len(all_stock_data) * 100
                st.metric("High Risk Stocks", f"{len(high_risk_stocks)}", f"{high_risk_pct:.1f}%")
            
            with risk_col3:
                risk_dispersion = np.std(portfolio_risks)
                st.metric("Risk Dispersion", f"{risk_dispersion:.1f}", help="Lower is better")
            
            with risk_col4:
                max_risk = max(portfolio_risks)
                st.metric("Maximum Risk", f"{max_risk:.0f}/100")
            
            # Risk distribution chart
            fig_risk_dist = px.histogram(
                x=portfolio_risks,
                nbins=10,
                title="Portfolio Risk Score Distribution",
                labels={'x': 'Risk Score', 'y': 'Number of Stocks'},
                color_discrete_sequence=['red']
            )
            fig_risk_dist.add_vline(x=np.mean(portfolio_risks), line_dash="dash", 
                                  annotation_text=f"Average: {np.mean(portfolio_risks):.0f}")
            fig_risk_dist.add_vrect(x0=0, x1=30, fillcolor="green", opacity=0.1, annotation_text="Low Risk")
            fig_risk_dist.add_vrect(x0=30, x1=60, fillcolor="yellow", opacity=0.1, annotation_text="Medium Risk")  
            fig_risk_dist.add_vrect(x0=60, x1=100, fillcolor="red", opacity=0.1, annotation_text="High Risk")
            
            st.plotly_chart(fig_risk_dist, use_container_width=True)
            
            # High risk stocks alert
            if high_risk_stocks:
                st.markdown("### ⚠️ High Risk Stocks Alert")
                
                for stock in high_risk_stocks:
                    with st.expander(f"🔴 {stock['symbol']} - Risk Score: {stock['risk_score']}/100"):
                        st.error("This stock has been flagged as high risk")
                        st.markdown("**Primary Risk Factors:**")
                        for factor in stock['factors']:
                            st.write(f"• {factor}")
            
            # Individual stock risk analysis
            st.markdown("### Individual Stock Risk Profiles")
            
            risk_stock = st.selectbox(
                "Select stock for detailed risk analysis:",
                options=list(all_stock_data.keys()),
                format_func=lambda x: f"{x.replace('.JK', '')} - {filtered_stocks.get(x, {}).get('name', x)}"
            )
            
            if risk_stock:
                stock_data = all_stock_data[risk_stock]
                metrics = analyzer.calculate_comprehensive_metrics(stock_data)
                
                # Create comprehensive risk chart
                risk_fig = create_risk_analysis_chart(metrics, filtered_stocks.get(risk_stock, {}))
                st.plotly_chart(risk_fig, use_container_width=True)
                
                # Risk factor breakdown
                risk_score, risk_factors = analyzer.comprehensive_risk_assessment(
                    metrics, filtered_stocks.get(risk_stock, {})
                )
                
                risk_detail_col1, risk_detail_col2 = st.columns(2)
                
                with risk_detail_col1:
                    st.markdown("**Risk Components**")
                    
                    # Financial risks
                    debt_risk = min(metrics.get('debt_to_equity', 0) / 50 * 100, 100)
                    liquidity_risk = max(0, 100 - metrics.get('current_ratio', 0) * 40)
                    
                    st.metric("Debt Risk", f"{debt_risk:.0f}/100")
                    st.metric("Liquidity Risk", f"{liquidity_risk:.0f}/100")
                    st.metric("Market Risk (Beta)", f"{metrics.get('beta', 1):.2f}")
                    st.metric("Volatility Risk", f"{metrics.get('max_drawdown', 0):.1f}%")
                
                with risk_detail_col2:
                    st.markdown("**Risk Mitigation**")
                    
                    if debt_risk > 60:
                        st.warning("Consider debt reduction plans")
                    if liquidity_risk > 50:
                        st.warning("Monitor cash flow closely")
                    if metrics.get('beta', 1) > 1.5:
                        st.info("Consider hedging strategies")
                    
                    st.markdown("**Portfolio Allocation**")
                    if risk_score < 30:
                        st.success("Suitable for core portfolio (up to 10%)")
                    elif risk_score < 60:
                        st.info("Moderate allocation recommended (up to 5%)")
                    else:
                        st.error("High risk - minimal allocation (<2%)")
        
        else:
            st.info("Risk analysis is disabled. Enable it in the sidebar to see comprehensive risk metrics.")
    
    with tab6:
        st.markdown("## Export & Reports")
        
        # Report generation options
        st.markdown("### Generate Reports")
        
        report_col1, report_col2, report_col3 = st.columns(3)
        
        with report_col1:
            if st.button("📊 Generate Executive Summary", type="primary"):
                # Create executive summary
                summary_data = []
                for result in analysis_results:
                    summary_data.append({
                        'Stock': result['symbol'].replace('.JK', ''),
                        'Company': result['company'][:30] + '...' if len(result['company']) > 30 else result['company'],
                        'Category': result['category'],
                        'Buffett Score': result['buffett_scores']['total_score'],
                        'Margin Safety': f"{result['avg_margin']:.1f}%",
                        'ML Return': f"{result['ml_return']:.1f}%" if result['ml_return'] else 'N/A',
                        'Recommendation': result['recommendation'],
                        'Risk Score': result['risk_score']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df = summary_df.sort_values('Buffett Score', ascending=False)
                
                st.success("Executive Summary Generated!")
                st.dataframe(summary_df, use_container_width=True)
                
                # Download as CSV
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"warren_buffett_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with report_col2:
            if st.button("🎯 Top Picks Report"):
                # Generate top picks based on composite scoring
                top_picks = sorted(analysis_results, key=lambda x: x['composite_score'], reverse=True)[:5]
                
                st.success("Top 5 Picks Generated!")
                
                for i, pick in enumerate(top_picks, 1):
                    with st.container():
                        st.markdown(f"### #{i} {pick['symbol'].replace('.JK', '')} - {pick['company']}")
                        
                        pick_col1, pick_col2, pick_col3 = st.columns(3)
                        
                        with pick_col1:
                            st.metric("Composite Score", f"{pick['composite_score']:.0f}/100")
                            st.metric("Buffett Score", f"{pick['buffett_scores']['total_score']}/100")
                        
                        with pick_col2:
                            st.metric("Margin Safety", f"{pick['avg_margin']:.1f}%")
                            st.metric("ML Return", f"{pick['ml_return']:.1f}%")
                        
                        with pick_col3:
                            st.metric("Risk Score", f"{pick['risk_score']}/100")
                            rec_color = "success" if "BUY" in pick['recommendation'] else "warning"
                            getattr(st, rec_color)(pick['recommendation'])
                        
                        st.markdown(f"**Investment Thesis:** {pick['reason']}")
                        st.markdown("---")
        
        with report_col3:
            if st.button("⚠️ Risk Report"):
                # Generate comprehensive risk report
                high_risk_stocks = [r for r in analysis_results if r['risk_score'] > 60]
                medium_risk_stocks = [r for r in analysis_results if 40 <= r['risk_score'] <= 60]
                low_risk_stocks = [r for r in analysis_results if r['risk_score'] < 40]
                
                st.success("Risk Report Generated!")
                
                st.markdown(f"**Portfolio Risk Distribution:**")
                st.markdown(f"• High Risk (60+): {len(high_risk_stocks)} stocks ({len(high_risk_stocks)/len(analysis_results)*100:.1f}%)")
                st.markdown(f"• Medium Risk (40-60): {len(medium_risk_stocks)} stocks ({len(medium_risk_stocks)/len(analysis_results)*100:.1f}%)")
                st.markdown(f"• Low Risk (<40): {len(low_risk_stocks)} stocks ({len(low_risk_stocks)/len(analysis_results)*100:.1f}%)")
                
                if high_risk_stocks:
                    st.markdown("**⚠️ High Risk Stocks:**")
                    for stock in high_risk_stocks:
                        st.error(f"• {stock['symbol'].replace('.JK', '')} - Risk Score: {stock['risk_score']}/100")
                        if stock['risk_factors']:
                            st.caption(f"  Main risks: {', '.join(stock['risk_factors'][:2])}")
        
        # Data export section
        st.markdown("### Export Data")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.markdown("**Available Export Formats:**")
            
            # Prepare comprehensive export data
            export_data = []
            for result in analysis_results:
                export_data.append({
                    'Symbol': result['symbol'],
                    'Company': result['company'],
                    'Category': result['category'], 
                    'Tier': result['tier'],
                    'Current_Price': result['current_price'],
                    'Buffett_Total_Score': result['buffett_scores']['total_score'],
                    'Value_Score': result['buffett_scores']['value_score'],
                    'Quality_Score': result['buffett_scores']['quality_score'],
                    'Growth_Score': result['buffett_scores']['growth_score'],
                    'Financial_Score': result['buffett_scores']['financial_score'],
                    'Management_Score': result['buffett_scores']['management_score'],
                    'PE_Ratio': result['metrics'].get('pe_ratio', 0),
                    'PB_Ratio': result['metrics'].get('pb_ratio', 0),
                    'ROE': result['metrics'].get('roe', 0),
                    'ROA': result['metrics'].get('roa', 0),
                    'Debt_Equity': result['metrics'].get('debt_to_equity', 0),
                    'Current_Ratio': result['metrics'].get('current_ratio', 0),
                    'Profit_Margin': result['metrics'].get('profit_margin', 0),
                    'Revenue_Growth': result['metrics'].get('revenue_growth', 0),
                    'Dividend_Yield': result['metrics'].get('dividend_yield', 0),
                    'Margin_Safety': result['avg_margin'],
                    'ML_Return_Prediction': result['ml_return'],
                    'Recommendation': result['recommendation'],
                    'Recommendation_Reason': result['reason'],
                    'Risk_Score': result['risk_score'],
                    'Composite_Score': result['composite_score']
                })
            
            export_df = pd.DataFrame(export_data)
            
            # CSV Export
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                csv_data,
                f"buffett_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                key='csv_download'
            )
            
            # Excel Export (if openpyxl available)
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='Analysis', index=False)
                    
                    # Add summary sheet
                    summary_data = {
                        'Metric': ['Total Stocks', 'High Quality (70+)', 'Undervalued (10%+)', 'Buy Recommendations'],
                        'Value': [
                            len(analysis_results),
                            len([r for r in analysis_results if r['buffett_scores']['total_score'] >= 70]),
                            len([r for r in analysis_results if r['avg_margin'] > 10]),
                            len([r for r in analysis_results if 'BUY' in r['recommendation']])
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                st.download_button(
                    "📊 Download Excel",
                    buffer.getvalue(),
                    f"buffett_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='excel_download'
                )
            except ImportError:
                st.info("Excel export requires openpyxl. Install with: pip install openpyxl")
        
        with export_col2:
            st.markdown("**Report Configuration:**")
            
            include_charts = st.checkbox("Include charts in report", value=True)
            include_technical = st.checkbox("Include technical analysis", value=use_technical)
            include_ml_details = st.checkbox("Include ML predictions", value=use_ml)
            
            # Generate PDF report (conceptual - would need reportlab)
            if st.button("📋 Generate PDF Report"):
                st.info("PDF report generation would require additional libraries (reportlab/weasyprint)")
                st.markdown("**Report would include:**")
                st.markdown("• Executive summary")
                st.markdown("• Individual stock analysis")
                st.markdown("• Portfolio risk assessment")
                st.markdown("• Investment recommendations")
                if include_charts:
                    st.markdown("• Performance charts and visualizations")
        
        # Model performance and statistics
        if use_ml and ml_results:
            st.markdown("### Model Performance Statistics")
            
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                st.markdown("**Cross-Validation Results:**")
                for timeframe, scores in ml_results['cv_scores'].items():
                    st.markdown(f"**{timeframe.upper()} Models:**")
                    for model, score in scores.items():
                        st.caption(f"• {model.upper()}: R² = {score:.3f}")
            
            with perf_col2:
                st.markdown("**Model Details:**")
                st.caption(f"Training samples: {ml_results['n_samples']}")
                st.caption(f"Features used: {ml_results['n_features_selected']}")
                st.caption(f"Best timeframe: {max(ml_results['cv_scores'].items(), key=lambda x: np.mean(list(x[1].values())))[0]}")
        
        # Analysis metadata
        st.markdown("### Analysis Metadata")
        
        metadata_col1, metadata_col2, metadata_col3 = st.columns(3)
        
        with metadata_col1:
            st.markdown("**Analysis Configuration:**")
            st.caption(f"• Mode: {analysis_mode}")
            st.caption(f"• ML Enabled: {use_ml}")
            st.caption(f"• Technical Analysis: {use_technical}")
            st.caption(f"• Risk Analysis: {include_risk_analysis}")
        
        with metadata_col2:
            st.markdown("**Filtering Criteria:**")
            st.caption(f"• Max P/E: {max_pe}")
            st.caption(f"• Max P/B: {max_pb}")
            st.caption(f"• Min ROE: {min_roe}%")
            st.caption(f"• Min Margin: {min_margin}%")
        
        with metadata_col3:
            st.markdown("**Data Sources:**")
            st.caption("• Yahoo Finance API")
            st.caption("• Real-time market data")
            st.caption(f"• Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.caption(f"• Total stocks: {len(all_stock_data)}")

    # Footer with disclaimer and additional info
    st.markdown("---")
    
    # Enhanced disclaimer
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
    <h4>⚠️ Investment Disclaimer</h4>
    <p><strong>This analysis is for educational and research purposes only.</strong></p>
    <ul>
    <li>Past performance does not guarantee future results</li>
    <li>All investments carry risk of loss</li>
    <li>Consult qualified financial advisors before making investment decisions</li>
    <li>Warren Buffett's principles are guidelines, not guarantees</li>
    <li>Machine learning predictions are estimates based on historical patterns</li>
    <li>Indonesian market conditions may differ from global markets</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics and credits
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Analysis Performance:**")
        total_time = time.time()  # Would need to track actual time
        st.caption(f"Stocks analyzed: {len(all_stock_data)}")
        st.caption(f"Data points processed: {len(all_stock_data) * 25}")  # Approximate
    
    with col2:
        st.markdown("**Methodology:**")
        st.caption("Enhanced Warren Buffett principles")
        st.caption("Multi-method intrinsic valuation")
        if use_ml:
            st.caption("Ensemble ML predictions")
        st.caption("Comprehensive risk assessment")
    
    with col3:
        st.markdown("**Version Info:**")
        st.caption("Warren Buffett Analyzer Pro v2.0")
        st.caption("Enhanced for Indonesian market")
        st.caption("Real-time data integration")
        st.caption("Advanced ML capabilities")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.markdown("Please refresh the page or contact support.")
        
        # Debug information (can be removed in production)
        if st.checkbox("Show debug info"):
            import traceback
            st.code(traceback.format_exc())
                
                
                