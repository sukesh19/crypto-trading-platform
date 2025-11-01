import requests
import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import hmac
import hashlib
import time
import json
import logging
from datetime import datetime
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoinDCXConfig:
    # CoinDCX API Configuration
    API_KEY = "YOUR_COINDCX_API_KEY"
    API_SECRET = "YOUR_COINDCX_API_SECRET"
    BASE_URL = "https://api.coindcx.com"
    
    # Trading Parameters
    TRADING_PAIRS = [
        'BTCUSDT',  # Bitcoin
        'ETHUSDT',  # Ethereum
        'BNBUSDT',  # Binance Coin
        'MATICUSDT', # Polygon
        'ADAUSDT',  # Cardano
        'SOLUSDT'   # Solana
    ]
    
    TIMEFRAME = '5m'  # 1m, 5m, 15m, 1h, 4h, 1d
    CANDLE_LIMIT = 500
    
    # Risk Management
    CAPITAL = 10000  # Starting capital in INR
    MAX_TRADES_PER_DAY = 15
    RISK_PERCENTAGE = 0.02  # 2% risk per trade
    MAX_POSITION_SIZE = 0.15  # 15% of portfolio per position
    
    STOP_LOSS_PERCENTAGE = 0.025  # 2.5%
    TAKE_PROFIT_PERCENTAGE = 0.05  # 5%
    TRAILING_STOP_PERCENTAGE = 0.02  # 2%
    ATR_MULTIPLIER = 2.0
    
    # Technical Indicators
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    EMA_SHORT = 9
    EMA_LONG = 21
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2
    
    # Filters
    MIN_VOLUME_24H = 1000000  # Minimum 24h volume in INR
    MAX_ATR_RATIO = 0.04  # Maximum ATR/Price ratio
    MIN_ADX = 20  # Minimum trend strength
    
    # Sentiment
    USE_SENTIMENT = True
    SENTIMENT_WEIGHT = 0.25

# --- CoinDCX API Client ---
class CoinDCXClient:
    def __init__(self, config):
        self.config = config
        self.base_url = config.BASE_URL
        self.api_key = config.API_KEY
        self.api_secret = config.API_SECRET.encode()
        
    def _generate_signature(self, payload):
        """Generate HMAC SHA256 signature for authenticated requests"""
        json_payload = json.dumps(payload, separators=(',', ':'))
        signature = hmac.new(self.api_secret, json_payload.encode(), hashlib.sha256).hexdigest()
        return signature
    
    def _make_request(self, endpoint, method='GET', data=None, auth=False):
        """Make HTTP request to CoinDCX API"""
        url = f"{self.base_url}{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        try:
            if auth and data:
                data['timestamp'] = int(time.time() * 1000)
                headers['X-AUTH-APIKEY'] = self.api_key
                headers['X-AUTH-SIGNATURE'] = self._generate_signature(data)
                response = requests.post(url, json=data, headers=headers, timeout=10)
            elif method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            else:
                response = requests.post(url, json=data, headers=headers, timeout=10)
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_ticker(self, market):
        """Get current ticker data for a market"""
        endpoint = f"/exchange/ticker"
        data = self._make_request(endpoint)
        if data:
            for ticker in data:
                if ticker.get('market') == market:
                    return ticker
        return None
    
    def get_candles(self, market, interval='5m', limit=500):
        """Get historical candlestick data"""
        # CoinDCX candles endpoint
        endpoint = f"/market_data/candles"
        params = {
            'pair': market,
            'interval': interval,
            'limit': limit
        }
        
        # Make request with params
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                df = pd.DataFrame(data)
                # CoinDCX format: [time, open, high, low, close, volume]
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Convert to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df.sort_values('timestamp').reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error fetching candles for {market}: {e}")
        
        return None
    
    def get_balance(self):
        """Get account balance"""
        endpoint = "/exchange/v1/users/balances"
        data = {}
        return self._make_request(endpoint, method='POST', data=data, auth=True)
    
    def get_order_book(self, market):
        """Get order book for a market"""
        endpoint = f"/market_data/orderbook"
        params = {'pair': market}
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=10)
            return response.json()
        except:
            return None
    
    def place_order(self, market, side, order_type, quantity, price=None):
        """
        Place order on CoinDCX
        side: 'buy' or 'sell'
        order_type: 'market_order' or 'limit_order'
        """
        endpoint = "/exchange/v1/orders/create"
        
        order_data = {
            'side': side,
            'order_type': order_type,
            'market': market,
            'total_quantity': quantity
        }
        
        if order_type == 'limit_order' and price:
            order_data['price_per_unit'] = price
        
        result = self._make_request(endpoint, method='POST', data=order_data, auth=True)
        return result
    
    def get_active_orders(self, market=None):
        """Get active orders"""
        endpoint = "/exchange/v1/orders/active_orders"
        data = {}
        if market:
            data['market'] = market
        
        return self._make_request(endpoint, method='POST', data=data, auth=True)
    
    def cancel_order(self, order_id):
        """Cancel an order"""
        endpoint = "/exchange/v1/orders/cancel"
        data = {'id': order_id}
        return self._make_request(endpoint, method='POST', data=data, auth=True)

# --- Technical Analysis ---
class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(df):
        """Calculate technical indicators"""
        if df is None or len(df) < 50:
            return None
        
        # Trend Indicators
        df['EMA_short'] = talib.EMA(df['close'], timeperiod=CoinDCXConfig.EMA_SHORT)
        df['EMA_long'] = talib.EMA(df['close'], timeperiod=CoinDCXConfig.EMA_LONG)
        df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
        
        # Momentum Indicators
        df['RSI'] = talib.RSI(df['close'], timeperiod=CoinDCXConfig.RSI_PERIOD)
        macd, signal, hist = talib.MACD(df['close'], 
                                        fastperiod=CoinDCXConfig.MACD_FAST,
                                        slowperiod=CoinDCXConfig.MACD_SLOW,
                                        signalperiod=CoinDCXConfig.MACD_SIGNAL)
        df['MACD'] = macd
        df['MACD_signal'] = signal
        df['MACD_hist'] = hist
        
        # Volatility Indicators
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        upper, middle, lower = talib.BBANDS(df['close'], 
                                            timeperiod=CoinDCXConfig.BB_PERIOD,
                                            nbdevup=CoinDCXConfig.BB_STD,
                                            nbdevdn=CoinDCXConfig.BB_STD)
        df['BB_upper'] = upper
        df['BB_middle'] = middle
        df['BB_lower'] = lower
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # Volume Indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        
        # Strength Indicators
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # Stochastic
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # Price patterns
        df['price_change_pct'] = df['close'].pct_change() * 100
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    @staticmethod
    def generate_signals(df):
        """Generate buy/sell signals from technical indicators"""
        signals = {
            'buy_score': 0,
            'sell_score': 0,
            'strength': 0
        }
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Trend Signals
        if latest['EMA_short'] > latest['EMA_long']:
            signals['buy_score'] += 1.5
        else:
            signals['sell_score'] += 1.5
        
        # EMA Crossover
        if prev['EMA_short'] <= prev['EMA_long'] and latest['EMA_short'] > latest['EMA_long']:
            signals['buy_score'] += 2
        elif prev['EMA_short'] >= prev['EMA_long'] and latest['EMA_short'] < latest['EMA_long']:
            signals['sell_score'] += 2
        
        # RSI Signals
        if latest['RSI'] < CoinDCXConfig.RSI_OVERSOLD:
            signals['buy_score'] += 2
        elif latest['RSI'] > CoinDCXConfig.RSI_OVERBOUGHT:
            signals['sell_score'] += 2
        elif 40 < latest['RSI'] < 60:
            signals['strength'] += 0.5
        
        # MACD Signals
        if latest['MACD'] > latest['MACD_signal'] and latest['MACD_hist'] > 0:
            signals['buy_score'] += 1.5
        elif latest['MACD'] < latest['MACD_signal'] and latest['MACD_hist'] < 0:
            signals['sell_score'] += 1.5
        
        # MACD Crossover
        if prev['MACD'] <= prev['MACD_signal'] and latest['MACD'] > latest['MACD_signal']:
            signals['buy_score'] += 2
        elif prev['MACD'] >= prev['MACD_signal'] and latest['MACD'] < latest['MACD_signal']:
            signals['sell_score'] += 2
        
        # Bollinger Bands
        if latest['close'] < latest['BB_lower']:
            signals['buy_score'] += 1.5
        elif latest['close'] > latest['BB_upper']:
            signals['sell_score'] += 1.5
        
        # ADX - Trend Strength
        if latest['ADX'] > 25:
            signals['strength'] += 1
        if latest['ADX'] > 40:
            signals['strength'] += 1
        
        # Stochastic
        if latest['STOCH_K'] < 20:
            signals['buy_score'] += 1
        elif latest['STOCH_K'] > 80:
            signals['sell_score'] += 1
        
        # MFI (Money Flow Index)
        if latest['MFI'] < 20:
            signals['buy_score'] += 1
        elif latest['MFI'] > 80:
            signals['sell_score'] += 1
        
        return signals

# --- Sentiment Analysis ---
class SentimentAnalyzer:
    @staticmethod
    def get_crypto_sentiment(symbol):
        """Get sentiment for crypto from news/social media"""
        try:
            # Extract base symbol (BTC from BTCUSDT)
            base_symbol = symbol.replace('USDT', '').replace('INR', '')
            
            # Simplified sentiment - in production, use CryptoPanic API or Twitter API
            # For demo, returning neutral sentiment
            sentiment_score = 0.0
            
            # You can integrate:
            # 1. CryptoPanic API
            # 2. Twitter API
            # 3. Reddit API
            # 4. News APIs
            
            logger.info(f"Sentiment for {base_symbol}: {sentiment_score}")
            return sentiment_score
        except Exception as e:
            logger.error(f"Error getting sentiment: {e}")
            return 0.0

# --- Risk Management ---
class RiskManager:
    def __init__(self, config):
        self.config = config
        self.trades_today = 0
        self.daily_pnl = 0
        self.max_drawdown = 0
    
    def calculate_position_size(self, balance, entry_price, stop_loss_price):
        """Calculate position size based on risk management"""
        risk_amount = balance * self.config.RISK_PERCENTAGE
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        max_position = (balance * self.config.MAX_POSITION_SIZE) / entry_price
        
        return min(position_size, max_position)
    
    def calculate_stop_loss(self, df, position_type='long'):
        """Calculate dynamic stop loss using ATR"""
        atr = df['ATR'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if position_type == 'long':
            atr_stop = current_price - (atr * self.config.ATR_MULTIPLIER)
            pct_stop = current_price * (1 - self.config.STOP_LOSS_PERCENTAGE)
            return max(atr_stop, pct_stop)
        else:
            atr_stop = current_price + (atr * self.config.ATR_MULTIPLIER)
            pct_stop = current_price * (1 + self.config.STOP_LOSS_PERCENTAGE)
            return min(atr_stop, pct_stop)
    
    def calculate_take_profit(self, entry_price, position_type='long'):
        """Calculate take profit level"""
        if position_type == 'long':
            return entry_price * (1 + self.config.TAKE_PROFIT_PERCENTAGE)
        else:
            return entry_price * (1 - self.config.TAKE_PROFIT_PERCENTAGE)
    
    def should_trade(self, df, ticker):
        """Check if trading conditions are favorable"""
        # Check trade limit
        if self.trades_today >= self.config.MAX_TRADES_PER_DAY:
            logger.warning("Daily trade limit reached")
            return False
        
        # Check volatility (ATR)
        atr_ratio = df['ATR'].iloc[-1] / df['close'].iloc[-1]
        if atr_ratio > self.config.MAX_ATR_RATIO:
            logger.warning(f"Too volatile: ATR ratio {atr_ratio:.4f}")
            return False
        
        # Check ADX (trend strength)
        if df['ADX'].iloc[-1] < self.config.MIN_ADX:
            logger.warning(f"Weak trend: ADX {df['ADX'].iloc[-1]:.2f}")
            return False
        
        # Check volume
        if df['volume'].iloc[-1] < df['volume_sma'].iloc[-1] * 0.5:
            logger.warning("Low volume")
            return False
        
        return True

# --- Main Trading Bot ---
class CoinDCXTradingBot:
    def __init__(self):
        self.config = CoinDCXConfig()
        self.client = CoinDCXClient(self.config)
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_manager = RiskManager(self.config)
        self.active_positions = {}
        self.portfolio_value = self.config.CAPITAL
    
    def get_portfolio_balance(self):
        """Get current portfolio balance"""
        balance = self.client.get_balance()
        if balance:
            usdt_balance = 0
            for asset in balance:
                if asset['currency'] == 'USDT':
                    usdt_balance = float(asset['balance'])
                    break
            return usdt_balance
        return self.portfolio_value
    
    def analyze_market(self, market):
        """Analyze a specific market"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing {market}")
        logger.info(f"{'='*50}")
        
        # Get market data
        df = self.client.get_candles(market, self.config.TIMEFRAME, self.config.CANDLE_LIMIT)
        if df is None or len(df) < 50:
            logger.error(f"Insufficient data for {market}")
            return None
        
        # Get current ticker
        ticker = self.client.get_ticker(market)
        if not ticker:
            logger.error(f"Could not fetch ticker for {market}")
            return None
        
        # Calculate indicators
        df = self.technical_analyzer.calculate_indicators(df)
        if df is None:
            return None
        
        # Generate signals
        signals = self.technical_analyzer.generate_signals(df)
        
        # Get sentiment
        sentiment_score = 0.0
        if self.config.USE_SENTIMENT:
            sentiment_score = self.sentiment_analyzer.get_crypto_sentiment(market)
        
        # Check trading conditions
        can_trade = self.risk_manager.should_trade(df, ticker)
        
        return {
            'df': df,
            'ticker': ticker,
            'signals': signals,
            'sentiment': sentiment_score,
            'can_trade': can_trade
        }
    
    def make_decision(self, market, analysis):
        """Make trading decision based on analysis"""
        signals = analysis['signals']
        sentiment = analysis['sentiment']
        
        # Combine technical and sentiment scores
        buy_score = signals['buy_score']
        sell_score = signals['sell_score']
        
        # Add sentiment weight
        if self.config.USE_SENTIMENT:
            if sentiment > 0:
                buy_score += sentiment * self.config.SENTIMENT_WEIGHT * 10
            else:
                sell_score += abs(sentiment) * self.config.SENTIMENT_WEIGHT * 10
        
        # Add trend strength bonus
        buy_score += signals['strength']
        sell_score += signals['strength']
        
        logger.info(f"Buy Score: {buy_score:.2f} | Sell Score: {sell_score:.2f}")
        
        # Decision threshold
        if buy_score >= 5 and buy_score > sell_score * 1.3:
            return 'BUY'
        elif sell_score >= 5 and sell_score > buy_score * 1.3:
            return 'SELL'
        
        return 'HOLD'
    
    def execute_trade(self, market, decision, analysis):
        """Execute trading decision"""
        if not analysis['can_trade']:
            logger.info("Trading conditions not met. Skipping.")
            return
        
        df = analysis['df']
        current_price = float(analysis['ticker']['last_price'])
        balance = self.get_portfolio_balance()
        
        logger.info(f"Decision: {decision} | Price: {current_price} | Balance: {balance:.2f} USDT")
        
        if decision == 'BUY' and market not in self.active_positions:
            # Calculate trade parameters
            stop_loss = self.risk_manager.calculate_stop_loss(df, 'long')
            take_profit = self.risk_manager.calculate_take_profit(current_price, 'long')
            position_size = self.risk_manager.calculate_position_size(balance, current_price, stop_loss)
            
            if position_size > 0:
                logger.info(f"Placing BUY order:")
                logger.info(f"  Size: {position_size:.6f}")
                logger.info(f"  Stop Loss: {stop_loss:.2f}")
                logger.info(f"  Take Profit: {take_profit:.2f}")
                
                # Place order (uncomment when ready to trade live)
                # order = self.client.place_order(market, 'buy', 'market_order', position_size)
                
                # For demo, just log
                self.active_positions[market] = {
                    'side': 'long',
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': position_size,
                    'entry_time': datetime.now()
                }
                self.risk_manager.trades_today += 1
                logger.info(f"✅ Position opened: {market}")
    
    def manage_positions(self):
        """Manage active positions"""
        for market, position in list(self.active_positions.items()):
            ticker = self.client.get_ticker(market)
            if not ticker:
                continue
            
            current_price = float(ticker['last_price'])
            entry_price = position['entry_price']
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            logger.info(f"\n📊 Position: {market}")
            logger.info(f"  Entry: {entry_price:.2f} | Current: {current_price:.2f}")
            logger.info(f"  PnL: {pnl_pct:+.2f}%")
            
            # Check stop loss
            if position['side'] == 'long' and current_price <= position['stop_loss']:
                logger.warning(f"🛑 Stop Loss Hit for {market}")
                self.close_position(market, 'Stop Loss')
            
            # Check take profit
            elif position['side'] == 'long' and current_price >= position['take_profit']:
                logger.info(f"🎯 Take Profit Hit for {market}")
                self.close_position(market, 'Take Profit')
    
    def close_position(self, market, reason):
        """Close a position"""
        if market in self.active_positions:
            position = self.active_positions[market]
            
            # Place sell order (uncomment for live trading)
            # self.client.place_order(market, 'sell', 'market_order', position['size'])
            
            logger.info(f"✅ Position closed: {market} - {reason}")
            del self.active_positions[market]
    
    def run(self):
        """Main bot loop"""
        logger.info("="*60)
        logger.info("🚀 CoinDCX Trading Bot Started")
        logger.info("="*60)
        logger.info(f"Trading Pairs: {', '.join(self.config.TRADING_PAIRS)}")
        logger.info(f"Timeframe: {self.config.TIMEFRAME}")
        logger.info(f"Max Trades/Day: {self.config.MAX_TRADES_PER_DAY}")
        logger.info("="*60)
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*60}")
                
                # Analyze each market
                for market in self.config.TRADING_PAIRS:
                    try:
                        analysis = self.analyze_market(market)
                        if analysis:
                            decision = self.make_decision(market, analysis)
                            if decision in ['BUY', 'SELL']:
                                self.execute_trade(market, decision, analysis)
                        
                        time.sleep(2)  # Avoid rate limits
                        
                    except Exception as e:
                        logger.error(f"Error analyzing {market}: {e}")
                
                # Manage existing positions
                if self.active_positions:
                    logger.info("\n📈 Managing Active Positions...")
                    self.manage_positions()
                else:
                    logger.info("\n📭 No active positions")
                
                # Reset daily counter at midnight
                current_hour = datetime.now().hour
                if current_hour == 0 and self.risk_manager.trades_today > 0:
                    self.risk_manager.trades_today = 0
                    logger.info("🔄 Daily trade counter reset")
                
                # Sleep before next iteration
                sleep_time = 300  # 5 minutes
                logger.info(f"\n💤 Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("\n🛑 Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)

# --- Entry Point ---
if __name__ == "__main__":
    bot = CoinDCXTradingBot()
    bot.run()
