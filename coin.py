import requests
import pandas as pd
import numpy as np
import hmac
import hashlib
import json
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bybit_trading_bot.log'),
        logging.StreamHandler()
    ]
)

class BybitTradingBot:
    def __init__(self, api_key, api_secret, test_mode=True, testnet=False):
        """
        Initialize the Bybit trading bot
        
        Args:
            api_key: Your Bybit API key
            api_secret: Your Bybit API secret
            test_mode: If True, runs in paper trading mode (no real orders)
            testnet: If True, uses Bybit testnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.test_mode = test_mode
        self.testnet = testnet
        
        # Set API URLs
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
        
        self.capital = 1.0  # Starting capital in USDT
        self.risk_per_trade = 0.5  # Risk 50% per trade
        
        self.symbol = "XRPUSDT"  # Bybit format
        self.category = "linear"  # Linear perpetual futures
        self.timeframe = '5'  # 5 minutes
        self.position = None
        self.bars_since_trade = 0
        
        # Trade tracking
        self.trades_log = []
        self.max_leverage = 50  # Bybit allows up to 100x for XRP, using 50x as safe max
        
        logging.info("="*60)
        logging.info("Bybit Trading Bot Initialized")
        logging.info(f"Mode: {'TESTNET' if testnet else 'MAINNET'} | {'PAPER TRADING' if test_mode else 'LIVE TRADING'}")
        logging.info(f"Symbol: {self.symbol}")
        logging.info(f"Base URL: {self.base_url}")
        logging.info("="*60)
    
    def _generate_signature(self, params):
        """Generate HMAC SHA256 signature for Bybit"""
        param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _send_request(self, endpoint, method="GET", params=None):
        """Send authenticated request to Bybit API"""
        if params is None:
            params = {}
        
        # Add timestamp and recv_window
        timestamp = str(int(time.time() * 1000))
        params['api_key'] = self.api_key
        params['timestamp'] = timestamp
        params['recv_window'] = '5000'
        
        # Generate signature
        signature = self._generate_signature(params)
        params['sign'] = signature
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, params=params, timeout=10)
            else:
                response = requests.post(url, data=params, timeout=10)
            
            data = response.json()
            
            if data.get('retCode') != 0:
                logging.error(f"API Error: {data.get('retMsg')}")
                return None
            
            return data.get('result')
            
        except Exception as e:
            logging.error(f"Request failed: {e}")
            return None
    
    def _send_public_request(self, endpoint, params=None):
        """Send public (non-authenticated) request"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('retCode') != 0:
                logging.error(f"API Error: {data.get('retMsg')}")
                return None
            
            return data.get('result')
            
        except Exception as e:
            logging.error(f"Request failed: {e}")
            return None
    
    def get_account_balance(self):
        """Fetch USDT balance from wallet"""
        try:
            result = self._send_request("/v5/account/wallet-balance", params={
                'accountType': 'UNIFIED'  # or 'CONTRACT' for derivatives only
            })
            
            if result and 'list' in result:
                for account in result['list']:
                    for coin in account.get('coin', []):
                        if coin['coin'] == 'USDT':
                            available = float(coin['availableToWithdraw'])
                            total = float(coin['walletBalance'])
                            logging.info(f"💰 USDT Balance: ${total:.2f} (Available: ${available:.2f})")
                            return available
            
            return self.capital
            
        except Exception as e:
            logging.error(f"Error fetching balance: {e}")
            return self.capital
    
    def get_current_price(self):
        """Fetch current market price"""
        try:
            result = self._send_public_request("/v5/market/tickers", params={
                'category': self.category,
                'symbol': self.symbol
            })
            
            if result and 'list' in result and len(result['list']) > 0:
                ticker = result['list'][0]
                price = float(ticker['lastPrice'])
                logging.debug(f"Current price: ${price:.4f}")
                return price
            
            return None
            
        except Exception as e:
            logging.error(f"Error fetching price: {e}")
            return None
    
    def get_recent_data(self, lookback=200):
        """Fetch recent candlestick data"""
        try:
            # Bybit intervals: 1,3,5,15,30,60,120,240,360,720,D,W,M
            result = self._send_public_request("/v5/market/kline", params={
                'category': self.category,
                'symbol': self.symbol,
                'interval': self.timeframe,
                'limit': lookback
            })
            
            if not result or 'list' not in result:
                logging.error("No kline data received")
                return None
            
            # Bybit returns: [startTime, open, high, low, close, volume, turnover]
            data = result['list']
            
            if len(data) == 0:
                logging.error("Empty kline data")
                return None
            
            logging.info(f"Received {len(data)} candles")
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            # Convert timestamp (milliseconds) to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms', utc = True)
            df.set_index('timestamp', inplace=True)
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Bybit returns newest first, so reverse
            df = df.sort_index()
            
            # Check data freshness
            latest_time = df.index[-1]
            current_time = pd.Timestamp.now(tz='UTC')
            time_diff = current_time - latest_time
            
            if time_diff.total_seconds() > 600:
                logging.warning(f"⚠️  Data age: {time_diff}")
            else:
                logging.info(f"✅ Data is fresh (age: {time_diff.total_seconds():.0f}s)")
            
            logging.info(f"Price range: ${df['close'].min():.4f} - ${df['close'].max():.4f}")
            logging.info(f"Latest close: ${df['close'].iloc[-1]:.4f}")
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logging.error(f"Error fetching kline data: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # RSI
        delta = df['close'].diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        df['rsi'] = 100 - 100 / (1 + up / (down + 1e-8))
        
        # EMAs
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        
        # Volume
        df['vol_ma'] = df['volume'].rolling(20).mean()
        
        # Donchian Channels
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['mid_20'] = (df['high_20'] + df['low_20']) / 2
        
        # Range metrics
        df['range'] = df['high_20'] - df['low_20']
        df['range_pct'] = df['range'] / df['close'] * 100
        
        return df.dropna()
    
    def calculate_leverage(self, price, sl_price, max_leverage=None):
        """Calculate leverage to risk exactly 50% of capital"""
        sl_distance_pct = abs(price - sl_price) / price
        ideal_leverage = self.risk_per_trade / sl_distance_pct
        ideal_leverage = round(ideal_leverage, 1)
        
        if max_leverage and ideal_leverage > max_leverage:
            logging.warning(f"⚠️  Calculated leverage {ideal_leverage}x exceeds max {max_leverage}x")
            leverage = max_leverage
            actual_risk = max_leverage * sl_distance_pct
            logging.warning(f"Actual risk: {(actual_risk * 100):.1f}%")
        else:
            leverage = ideal_leverage
            actual_risk = self.risk_per_trade
        
        position_size = (self.capital * leverage) / price
        return leverage, position_size, actual_risk
    
    def set_leverage(self, leverage):
        """Set leverage for the symbol"""
        try:
            params = {
                'category': self.category,
                'symbol': self.symbol,
                'buyLeverage': str(leverage),
                'sellLeverage': str(leverage)
            }
            
            result = self._send_request("/v5/position/set-leverage", method="POST", params=params)
            
            if result:
                logging.info(f"✅ Leverage set to {leverage}x")
                return True
            else:
                logging.error(f"Failed to set leverage")
                return False
                
        except Exception as e:
            logging.error(f"Error setting leverage: {e}")
            return False
    
    def place_order(self, side, price, sl, tp, size, leverage):
        """Place a futures order with stop loss and take profit"""
        if self.test_mode:
            logging.info(f"[TEST MODE] Would place {side} order:")
            logging.info(f"  Entry: ${price:.4f} | Size: {size:.4f} | Leverage: {leverage}x")
            logging.info(f"  SL: ${sl:.4f} | TP: ${tp:.4f}")
            return True
        
        try:
            # Set leverage first
            self.set_leverage(int(leverage))
            
            # Place market order
            params = {
                'category': self.category,
                'symbol': self.symbol,
                'side': 'Buy' if side == 'buy' else 'Sell',
                'orderType': 'Market',
                'qty': str(round(size, 2)),
                'stopLoss': str(round(sl, 4)),
                'takeProfit': str(round(tp, 4)),
                'positionIdx': 0  # One-way mode
            }
            
            result = self._send_request("/v5/order/create", method="POST", params=params)
            
            if result:
                order_id = result.get('orderId')
                logging.info(f"✅ Order placed successfully: {order_id}")
                logging.info(f"   {side.upper()} | Entry: ${price:.4f} | Leverage: {leverage}x")
                logging.info(f"   SL: ${sl:.4f} | TP: ${tp:.4f}")
                return True
            else:
                logging.error("Order placement failed")
                return False
                
        except Exception as e:
            logging.error(f"Error placing order: {e}")
            return False
    
    def get_position(self):
        """Get current position for the symbol"""
        try:
            params = {
                'category': self.category,
                'symbol': self.symbol
            }
            
            result = self._send_request("/v5/position/list", params=params)
            
            if result and 'list' in result and len(result['list']) > 0:
                pos = result['list'][0]
                size = float(pos['size'])
                
                if size > 0:
                    return {
                        'side': pos['side'].lower(),
                        'size': size,
                        'entry': float(pos['avgPrice']),
                        'unrealizedPnl': float(pos['unrealisedPnl']),
                        'leverage': float(pos['leverage'])
                    }
            
            return None
            
        except Exception as e:
            logging.error(f"Error fetching position: {e}")
            return None
    
    def close_position(self):
        """Close the current position"""
        if not self.position:
            return
        
        if not self.test_mode:
            try:
                # Close via market order in opposite direction
                side = 'Sell' if self.position['side'] == 'buy' else 'Buy'
                
                params = {
                    'category': self.category,
                    'symbol': self.symbol,
                    'side': side,
                    'orderType': 'Market',
                    'qty': str(self.position['size']),
                    'reduceOnly': True,
                    'positionIdx': 0
                }
                
                result = self._send_request("/v5/order/create", method="POST", params=params)
                
                if result:
                    logging.info(f"✅ Position closed")
                    
            except Exception as e:
                logging.error(f"Error closing position: {e}")
        
        logging.info(f"Closing {self.position['side']} position")
        self.position = None
        self.bars_since_trade = 0
    
    def check_exit_conditions(self, current_price):
        """Check if position should be closed"""
        if not self.position:
            return False
        
        side = self.position['side']
        sl = self.position['sl']
        tp = self.position['tp']
        
        # Check stop loss
        if side == 'buy' and current_price <= sl:
            logging.info(f"🛑 Stop loss hit! Price: ${current_price:.4f} <= SL: ${sl:.4f}")
            return True
        elif side == 'sell' and current_price >= sl:
            logging.info(f"🛑 Stop loss hit! Price: ${current_price:.4f} >= SL: ${sl:.4f}")
            return True
        
        # Check take profit
        if side == 'buy' and current_price >= tp:
            logging.info(f"🎯 Take profit hit! Price: ${current_price:.4f} >= TP: ${tp:.4f}")
            return True
        elif side == 'sell' and current_price <= tp:
            logging.info(f"🎯 Take profit hit! Price: ${current_price:.4f} <= TP: ${tp:.4f}")
            return True
        
        return False
    
    def check_entry_signals(self, df):
        """Check for entry signals - inverted breakout strategy"""
        if len(df) < 50:
            return None
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        price = current['close']
        atr = current['atr']
        
        # Skip if not enough bars since last trade
        if self.bars_since_trade < 4:
            return None
        
        # Skip during low liquidity hours (UTC 0-6)
        utc_hour = current.name.hour
        if utc_hour in [0, 1, 2, 3, 4, 5, 6]:
            logging.debug("Skipping: Low liquidity hours")
            return None
        
        # Check if range is tight
        range_tight = current['range_pct'] < 3.5
        if not range_tight:
            logging.debug(f"Range too wide: {current['range_pct']:.2f}%")
            return None
        
        # INVERTED STRATEGY: Fade breakouts
        
        # Fade upside breakout -> GO SHORT
        if (current['high'] > prev['high_20'] and
            prev['high'] <= prev['high_20'] and
            current['volume'] > current['vol_ma'] * 1.5 and
            current['rsi'] > 50 and current['rsi'] < 70 and
            current['close'] > current['ema20'] and
            current['ema20'] > current['ema50'] and
            current['close'] > current['open']):
            
            sl = price + 2.5 * atr
            tp = price - 1.5 * atr
            leverage, size, actual_risk = self.calculate_leverage(price, sl, self.max_leverage)
            
            return {
                'side': 'sell',
                'entry': price,
                'sl': sl,
                'tp': tp,
                'size': size,
                'leverage': leverage,
                'actual_risk': actual_risk,
                'reason': '📉 Fade upside breakout'
            }
        
        # Fade downside breakdown -> GO LONG
        elif (current['low'] < prev['low_20'] and
              prev['low'] >= prev['low_20'] and
              current['volume'] > current['vol_ma'] * 1.5 and
              current['rsi'] < 50 and current['rsi'] > 30 and
              current['close'] < current['ema20'] and
              current['ema20'] < current['ema50'] and
              current['close'] < current['open']):
            
            sl = price - 2.5 * atr
            tp = price + 1.5 * atr
            leverage, size, actual_risk = self.calculate_leverage(price, sl, self.max_leverage)
            
            return {
                'side': 'buy',
                'entry': price,
                'sl': sl,
                'tp': tp,
                'size': size,
                'leverage': leverage,
                'actual_risk': actual_risk,
                'reason': '📈 Fade downside breakdown'
            }
        
        return None
    
    def run(self):
        """Main trading loop"""
        logging.info("\n" + "="*60)
        logging.info("🚀 BYBIT TRADING BOT STARTED")
        logging.info(f"Symbol: {self.symbol}")
        logging.info(f"Timeframe: {self.timeframe}m")
        logging.info(f"Initial Capital: ${self.capital}")
        logging.info(f"Risk Per Trade: {self.risk_per_trade*100}%")
        logging.info(f"Max Leverage: {self.max_leverage}x")
        logging.info("="*60 + "\n")
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                logging.info(f"\n{'='*60}")
                logging.info(f"📊 Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logging.info(f"{'='*60}")
                
                # Update capital
                if not self.test_mode:
                    self.capital = self.get_account_balance()
                
                # Fetch data
                df = self.get_recent_data(lookback=200)
                if df is None or len(df) < 50:
                    logging.warning("Insufficient data, retrying in 60s...")
                    time.sleep(60)
                    continue
                
                # Calculate indicators
                df = self.calculate_indicators(df)
                current_price = self.get_current_price()
                
                if not current_price:
                    current_price = df.iloc[-1]['close']
                
                logging.info(f"💵 Current Price: ${current_price:.4f}")
                
                # Check positions
                if self.position:
                    if self.check_exit_conditions(current_price):
                        self.close_position()
                    else:
                        pnl_pct = 0
                        if self.position['side'] == 'buy':
                            pnl_pct = ((current_price - self.position['entry']) / self.position['entry']) * 100 * self.position['leverage']
                        else:
                            pnl_pct = ((self.position['entry'] - current_price) / self.position['entry']) * 100 * self.position['leverage']
                        
                        logging.info(f"📍 Position: {self.position['side'].upper()} | "
                                   f"Entry: ${self.position['entry']:.4f} | "
                                   f"P&L: {pnl_pct:+.2f}%")
                        logging.info(f"   SL: ${self.position['sl']:.4f} | TP: ${self.position['tp']:.4f}")
                
                # Check entry
                if not self.position:
                    signal = self.check_entry_signals(df)
                    if signal:
                        logging.info(f"\n🚨 SIGNAL: {signal['reason']}")
                        logging.info(f"Risk: {signal['actual_risk']*100:.1f}% | Leverage: {signal['leverage']:.1f}x")
                        
                        success = self.place_order(
                            signal['side'],
                            signal['entry'],
                            signal['sl'],
                            signal['tp'],
                            signal['size'],
                            signal['leverage']
                        )
                        
                        if success:
                            self.position = signal
                            self.bars_since_trade = 0
                            logging.info("✅ Position opened\n")
                    else:
                        self.bars_since_trade += 1
                        logging.info(f"⏳ No position | Bars since trade: {self.bars_since_trade}")
                
                # Sleep until next candle
                logging.info(f"\n💤 Sleeping 5 minutes...\n")
                time.sleep(300)
                
            except KeyboardInterrupt:
                logging.info("\n⚠️  Bot stopped by user")
                break
            except Exception as e:
                logging.error(f"❌ Error in main loop: {e}")
                import traceback
                logging.error(traceback.format_exc())
                time.sleep(60)

if __name__ == "__main__":
    # Get API credentials from environment variables (recommended)
    import os
    
    API_KEY = os.getenv('BYBIT_API_KEY')
    API_SECRET = os.getenv('BYBIT_API_SECRET')
    
    # Initialize bot
    bot = BybitTradingBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        test_mode=False,  # Set to False for live trading
        testnet=False    # Set to True to use testnet
    )
    
    # Run the bot
    bot.run()
