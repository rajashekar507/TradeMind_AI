"""
TradeMind_AI: Multi-Timeframe Market Analyzer
Analyzes NIFTY/BANKNIFTY across multiple timeframes
Working implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple
import logging

class MultiTimeframeAnalyzer:
    """Analyzes market data across multiple timeframes"""
    
    def __init__(self):
        """Initialize the analyzer"""
        self.logger = logging.getLogger('MultiTimeframeAnalyzer')
        
        # Timeframes to analyze
        self.timeframes = {
            '5m': {'period': '1d', 'interval': '5m'},
            '15m': {'period': '5d', 'interval': '15m'},
            '1h': {'period': '1mo', 'interval': '1h'},
            '1d': {'period': '3mo', 'interval': '1d'}
        }
        
        # Technical indicators settings
        self.ma_periods = {
            'fast': 9,
            'medium': 21,
            'slow': 50
        }
        
        self.logger.info("Multi-timeframe analyzer initialized")
    
    def fetch_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch market data for given symbol and timeframe"""
        try:
            # Convert symbol for yfinance
            ticker = "^NSEI" if symbol == "NIFTY" else "^NSEBANK"
            
            # Get timeframe config
            config = self.timeframes.get(timeframe, self.timeframes['5m'])
            
            # Fetch data
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=config['period'], interval=config['interval'])
            
            if data.empty:
                # Return sample data if yfinance fails
                return self._generate_sample_data(symbol, timeframe)
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Error fetching data: {e}. Using sample data.")
            return self._generate_sample_data(symbol, timeframe)
    
    def _generate_sample_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Generate sample data for testing"""
        base_price = 25000 if symbol == "NIFTY" else 52000
        
        # Number of candles based on timeframe
        candles = {
            '5m': 78,    # ~6.5 hours
            '15m': 100,  # ~25 hours
            '1h': 120,   # 5 days
            '1d': 90     # 3 months
        }.get(timeframe, 100)
        
        # Generate price data
        dates = pd.date_range(end=datetime.now(), periods=candles, freq=timeframe)
        prices = []
        
        current_price = base_price
        for _ in range(candles):
            change = np.random.randn() * base_price * 0.001  # 0.1% volatility
            current_price += change
            
            high = current_price + abs(np.random.randn() * base_price * 0.0005)
            low = current_price - abs(np.random.randn() * base_price * 0.0005)
            open_price = current_price + np.random.randn() * base_price * 0.0002
            
            prices.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': current_price,
                'Volume': int(1000000 * (1 + np.random.rand()))
            })
        
        df = pd.DataFrame(prices, index=dates)
        return df
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = data.copy()
        
        # Moving averages
        df['MA_fast'] = df['Close'].rolling(window=self.ma_periods['fast']).mean()
        df['MA_medium'] = df['Close'].rolling(window=self.ma_periods['medium']).mean()
        df['MA_slow'] = df['Close'].rolling(window=self.ma_periods['slow']).mean()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = self.calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def analyze_timeframe(self, symbol: str, timeframe: str) -> Dict:
        """Analyze a single timeframe"""
        # Fetch data
        data = self.fetch_data(symbol, timeframe)
        
        # Calculate indicators
        data = self.calculate_indicators(data)
        
        # Get latest values
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Determine signals
        signals = {
            'timeframe': timeframe,
            'price': latest['Close'],
            'change': latest['Close'] - prev['Close'],
            'change_pct': ((latest['Close'] - prev['Close']) / prev['Close']) * 100,
            'volume': latest['Volume'],
            'volume_ratio': latest.get('Volume_ratio', 1.0),
            'rsi': latest.get('RSI', 50),
            'macd_histogram': latest.get('MACD_hist', 0),
            'ma_alignment': self._check_ma_alignment(latest),
            'bb_position': self._check_bb_position(latest),
            'trend': self._determine_trend(data),
            'signal': self._generate_signal(latest, data)
        }
        
        return signals
    
    def _check_ma_alignment(self, latest: pd.Series) -> str:
        """Check moving average alignment"""
        if pd.isna(latest.get('MA_slow')):
            return 'NEUTRAL'
            
        if latest['MA_fast'] > latest['MA_medium'] > latest['MA_slow']:
            return 'BULLISH'
        elif latest['MA_fast'] < latest['MA_medium'] < latest['MA_slow']:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _check_bb_position(self, latest: pd.Series) -> str:
        """Check position relative to Bollinger Bands"""
        if pd.isna(latest.get('BB_upper')):
            return 'MIDDLE'
            
        if latest['Close'] > latest['BB_upper']:
            return 'ABOVE_UPPER'
        elif latest['Close'] < latest['BB_lower']:
            return 'BELOW_LOWER'
        else:
            return 'MIDDLE'
    
    def _determine_trend(self, data: pd.DataFrame) -> str:
        """Determine overall trend"""
        if len(data) < 20:
            return 'NEUTRAL'
            
        # Check last 20 candles
        recent = data.tail(20)
        start_price = recent.iloc[0]['Close']
        end_price = recent.iloc[-1]['Close']
        
        change_pct = ((end_price - start_price) / start_price) * 100
        
        if change_pct > 1:
            return 'UPTREND'
        elif change_pct < -1:
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'
    
    def _generate_signal(self, latest: pd.Series, data: pd.DataFrame) -> str:
        """Generate trading signal"""
        signals = []
        
        # RSI signals
        rsi = latest.get('RSI', 50)
        if rsi > 70:
            signals.append('OVERBOUGHT')
        elif rsi < 30:
            signals.append('OVERSOLD')
        
        # MA signals
        ma_alignment = self._check_ma_alignment(latest)
        if ma_alignment == 'BULLISH':
            signals.append('BUY')
        elif ma_alignment == 'BEARISH':
            signals.append('SELL')
        
        # MACD signals
        macd_hist = latest.get('MACD_hist', 0)
        if macd_hist > 0 and data['MACD_hist'].iloc[-2] <= 0:
            signals.append('BUY')
        elif macd_hist < 0 and data['MACD_hist'].iloc[-2] >= 0:
            signals.append('SELL')
        
        # Aggregate signals
        buy_signals = sum(1 for s in signals if s == 'BUY')
        sell_signals = sum(1 for s in signals if s == 'SELL')
        
        if buy_signals > sell_signals:
            return 'BUY'
        elif sell_signals > buy_signals:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def analyze_all_timeframes(self, symbol: str) -> Dict:
        """Analyze all timeframes and provide summary"""
        results = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'timeframes': {},
            'summary': {}
        }
        
        # Analyze each timeframe
        for tf in self.timeframes:
            results['timeframes'][tf] = self.analyze_timeframe(symbol, tf)
        
        # Generate summary
        signals = [tf['signal'] for tf in results['timeframes'].values()]
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        
        # Overall signal
        if buy_count > sell_count:
            overall_signal = 'BUY'
            confidence = (buy_count / len(signals)) * 100
        elif sell_count > buy_count:
            overall_signal = 'SELL'
            confidence = (sell_count / len(signals)) * 100
        else:
            overall_signal = 'NEUTRAL'
            confidence = 50
        
        results['summary'] = {
            'overall_signal': overall_signal,
            'confidence': confidence,
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'neutral_signals': signals.count('NEUTRAL')
        }
        
        return results
    
    def display_analysis(self, analysis: Dict):
        """Display analysis results"""
        print(f"\n{'='*60}")
        print(f"📊 MULTI-TIMEFRAME ANALYSIS - {analysis['symbol']}")
        print(f"{'='*60}")
        
        # Display each timeframe
        for tf, data in analysis['timeframes'].items():
            print(f"\n⏰ {tf} Timeframe:")
            print(f"   Price: ₹{data['price']:.2f} ({data['change_pct']:+.2f}%)")
            print(f"   RSI: {data['rsi']:.2f}")
            print(f"   Trend: {data['trend']}")
            print(f"   Signal: {data['signal']}")
            print(f"   MA Alignment: {data['ma_alignment']}")
        
        # Display summary
        summary = analysis['summary']
        print(f"\n🎯 SUMMARY:")
        print(f"   Overall Signal: {summary['overall_signal']}")
        print(f"   Confidence: {summary['confidence']:.1f}%")
        print(f"   Buy Signals: {summary['buy_signals']}")
        print(f"   Sell Signals: {summary['sell_signals']}")
        print(f"{'='*60}")

# Test the analyzer
if __name__ == "__main__":
    analyzer = MultiTimeframeAnalyzer()
    
    # Test with NIFTY
    print("Testing Multi-Timeframe Analyzer...")
    analysis = analyzer.analyze_all_timeframes('NIFTY')
    analyzer.display_analysis(analysis)
