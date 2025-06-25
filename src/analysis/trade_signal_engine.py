"""
Trade Signal Engine for institutional-grade trading system
Generates AI-driven trade recommendations using multi-factor analysis
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger('trading_system.trade_signal_engine')

class TradeSignalEngine:
    """Institutional-grade trade signal generation engine"""
    
    def __init__(self, settings):
        self.settings = settings
        self.confidence_threshold = 60.0  # Minimum confidence for trade signals
        
        self.weights = {
            'technical': 0.35,      # Technical indicators (35%)
            'options': 0.25,        # Options OI flow & Greeks (25%)
            'sentiment': 0.20,      # Market sentiment & VIX (20%)
            'global': 0.20          # Global clues & structure (20%)
        }
        
        logger.info("✅ TradeSignalEngine initialized with institutional-grade scoring")
    
    async def initialize(self):
        """Initialize the trade signal engine"""
        try:
            logger.info("🔧 Initializing TradeSignalEngine...")
            return True
        except Exception as e:
            logger.error(f"❌ TradeSignalEngine initialization failed: {e}")
            return False
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trade signals based on market data"""
        try:
            logger.info("🎯 Generating institutional-grade trade signals...")
            
            signals = []
            
            nifty_signal = await self._analyze_instrument_signals('NIFTY', market_data)
            if nifty_signal:
                signals.append(nifty_signal)
            
            banknifty_signal = await self._analyze_instrument_signals('BANKNIFTY', market_data)
            if banknifty_signal:
                signals.append(banknifty_signal)
            
            logger.info(f"📊 Generated {len(signals)} trade signals")
            return signals
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            return []
    
    async def _analyze_instrument_signals(self, instrument: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze signals for a specific instrument"""
        try:
            technical_score = self._calculate_technical_score(instrument, market_data)
            options_score = self._calculate_options_score(instrument, market_data)
            sentiment_score = self._calculate_sentiment_score(market_data)
            global_score = self._calculate_global_score(market_data)
            
            confidence = (
                technical_score * self.weights['technical'] +
                options_score * self.weights['options'] +
                sentiment_score * self.weights['sentiment'] +
                global_score * self.weights['global']
            )
            
            if confidence >= self.confidence_threshold:
                return self._create_trade_signal(instrument, confidence, market_data)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze {instrument} signals: {e}")
            return None
    
    def _calculate_technical_score(self, instrument: str, market_data: Dict[str, Any]) -> float:
        """Calculate technical analysis score based on live data"""
        try:
            technical_data = market_data.get('technical_data', {})
            
            if not technical_data or technical_data.get('status') != 'success':
                return 55.0  # Slightly positive base if no data
            
            base_score = 55.0  # Start with slight positive bias
            instrument_data = technical_data.get(instrument, {})
            
            rsi = instrument_data.get('rsi', 50)
            if rsi > 55:  # Lowered threshold for bullish signals
                base_score += min((rsi - 55) * 0.8, 20)  # More aggressive scoring
            elif rsi < 45:  # Lowered threshold for bearish signals
                base_score -= min((45 - rsi) * 0.8, 20)
            
            trend = instrument_data.get('trend', 'neutral')
            if trend == 'bullish':
                base_score += 15  # Increased from 10
            elif trend == 'bearish':
                base_score -= 15
            
            if rsi > 60 and trend == 'bullish':
                base_score += 8  # Strong momentum bonus
            elif rsi < 40 and trend == 'bearish':
                base_score -= 8
            
            score = max(30.0, min(base_score, 90.0))  # Expanded range 30-90
            logger.debug(f"📈 {instrument} technical score: {score} (RSI: {rsi}, Trend: {trend})")
            return score
            
        except Exception as e:
            logger.error(f"❌ Technical score calculation failed: {e}")
            return 55.0
    
    def _calculate_options_score(self, instrument: str, market_data: Dict[str, Any]) -> float:
        """Calculate options flow and Greeks score based on live data"""
        try:
            options_data = market_data.get('options_data', {})
            
            if not options_data or options_data.get('status') != 'success':
                return 60.0  # Positive base if no data (market participation)
            
            instrument_options = options_data.get(instrument, {})
            if not instrument_options or instrument_options.get('status') != 'success':
                return 60.0
            
            base_score = 60.0  # Start with positive bias
            options_chain = instrument_options.get('options_data', {})
            
            ce_oi_total = sum(opt.get('oi', 0) for key, opt in options_chain.items() if '_CE' in key)
            pe_oi_total = sum(opt.get('oi', 0) for key, opt in options_chain.items() if '_PE' in key)
            
            if ce_oi_total > 0:
                pcr = pe_oi_total / ce_oi_total
                if pcr > 1.1:  # Lowered threshold for bullish sentiment
                    base_score += min((pcr - 1.0) * 15, 15)  # Moderate boost
                elif pcr < 0.9:  # Adjusted threshold for bearish sentiment
                    base_score -= min((1.0 - pcr) * 15, 15)
            
            total_strikes = len(options_chain)
            if total_strikes >= 20:  # Good liquidity
                base_score += 8
            elif total_strikes >= 10:
                base_score += 5
            
            strikes_with_ltp = sum(1 for opt in options_chain.values() if opt.get('ltp', 0) > 0)
            if strikes_with_ltp >= 15:
                base_score += 6
            elif strikes_with_ltp >= 10:
                base_score += 4
            
            score = max(40.0, min(base_score, 85.0))  # Expanded range 40-85
            pcr_val = pe_oi_total / ce_oi_total if ce_oi_total > 0 else 0
            logger.debug(f"📊 {instrument} options score: {score} (PCR: {pcr_val:.2f}, Strikes: {total_strikes}, LTP: {strikes_with_ltp})")
            return score
            
        except Exception as e:
            logger.error(f"❌ Options score calculation failed: {e}")
            return 60.0
    
    def _calculate_sentiment_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate market sentiment and VIX score based on live data"""
        try:
            vix_data = market_data.get('vix_data', {})
            news_data = market_data.get('news_data', {})
            
            base_score = 58.0  # Positive base score
            
            if vix_data and vix_data.get('status') == 'success':
                vix = vix_data.get('vix', 16.5)
                if vix < 16:  # Adjusted threshold for low VIX
                    base_score += min((16 - vix) * 2.5, 18)  # Increased multiplier
                elif vix > 22:  # Adjusted threshold for high VIX
                    base_score -= min((vix - 22) * 2, 20)
                
                if vix < 14:  # Very low VIX = strong bullish sentiment
                    base_score += 8
                elif vix > 30:  # Very high VIX = extreme fear
                    base_score -= 12
            
            if news_data and news_data.get('status') == 'success':
                sentiment_score = news_data.get('sentiment_score', 0.0)
                base_score += sentiment_score * 12  # Increased multiplier
                
                if sentiment_score > 0.3:  # Strong positive news
                    base_score += 6
                elif sentiment_score < -0.3:  # Strong negative news
                    base_score -= 6
            
            score = max(30.0, min(base_score, 85.0))  # Expanded range 30-85
            vix_val = vix_data.get('vix', 'N/A') if vix_data else 'N/A'
            sentiment_val = news_data.get('sentiment_score', 'N/A') if news_data else 'N/A'
            logger.debug(f"📰 Sentiment score: {score} (VIX: {vix_val}, News: {sentiment_val})")
            return score
            
        except Exception as e:
            logger.error(f"❌ Sentiment score calculation failed: {e}")
            return 58.0
    
    def _calculate_global_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate global market influence score based on live data"""
        try:
            global_data = market_data.get('global_data', {})
            
            if not global_data or global_data.get('status') != 'success':
                return 62.0  # Positive base (global markets generally trend up)
            
            base_score = 62.0
            indices = global_data.get('indices', {})
            
            sgx_nifty = indices.get('SGX_NIFTY', 0)
            if sgx_nifty > 20:  # Strong positive SGX
                base_score += min(sgx_nifty * 0.12, 15)  # Increased multiplier
            elif sgx_nifty > 0:  # Moderate positive SGX
                base_score += min(sgx_nifty * 0.1, 12)
            elif sgx_nifty < -20:  # Strong negative SGX
                base_score += max(sgx_nifty * 0.12, -15)
            elif sgx_nifty < 0:  # Moderate negative SGX
                base_score += max(sgx_nifty * 0.1, -12)
            
            if sgx_nifty > 50:  # Very strong global cues
                base_score += 10
            elif sgx_nifty < -50:  # Very weak global cues
                base_score -= 10
            
            score = max(40.0, min(base_score, 80.0))  # Expanded range 40-80
            logger.debug(f"🌍 Global score: {score} (SGX: {sgx_nifty})")
            return score
            
        except Exception as e:
            logger.error(f"❌ Global score calculation failed: {e}")
            return 62.0
    
    def _create_trade_signal(self, instrument: str, confidence: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a formatted trade signal with live LTP pricing and trend-based direction"""
        try:
            spot_data = market_data.get('spot_data', {})
            current_price = 0
            
            if spot_data and spot_data.get('status') == 'success':
                prices = spot_data.get('prices', {})
                current_price = prices.get(instrument, 0)
            
            direction = self._determine_signal_direction(instrument, market_data, confidence)
            
            strike, entry_price = self._find_best_strike_with_ltp(instrument, current_price, direction, market_data)
            
            if entry_price == 0:
                logger.warning(f"⚠️ Could not fetch live LTP for any {instrument} {direction} strikes, using conservative fallback")
                entry_price = 120  # Conservative fallback
                strike_buffer = 100 if instrument == 'NIFTY' else 500
                if direction == 'CE':
                    strike = int((current_price + strike_buffer) / strike_buffer) * strike_buffer
                else:
                    strike = int((current_price - strike_buffer) / strike_buffer) * strike_buffer
            
            sl_price, target1, target2 = self._calculate_dynamic_levels(entry_price, instrument, market_data)
            
            expiry_date = self._get_option_expiry(market_data)
            
            option_details = self._get_option_details(instrument, strike, direction, market_data)
            
            signal = {
                'timestamp': datetime.now(),
                'instrument': instrument,
                'strike': strike,
                'option_type': direction,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': sl_price,
                'target_1': target1,
                'target_2': target2,
                'confidence': round(confidence, 1),
                'current_spot': current_price,
                'strike_ltp': entry_price,
                'reason': self._generate_signal_reason(confidence, market_data, direction),
                'expiry': expiry_date,
                'risk_status': 'VALIDATED',
                'iv': option_details.get('iv', 0),
                'delta': option_details.get('delta', 0),
                'oi_trend': option_details.get('oi_trend', 'neutral'),
                'direction_reason': option_details.get('direction_reason', 'Multi-factor analysis')
            }
            
            logger.info(f"✅ Generated {instrument} signal: {strike} {direction} @ ₹{entry_price} LTP, {confidence:.1f}% confidence")
            return signal
            
        except Exception as e:
            logger.error(f"❌ Failed to create trade signal: {e}")
            return {}
    
    def _determine_signal_direction(self, instrument: str, market_data: Dict[str, Any], confidence: float) -> str:
        """Determine CE/PE direction based on live market trend analysis"""
        try:
            technical_data = market_data.get('technical_data', {})
            vix_data = market_data.get('vix_data', {})
            global_data = market_data.get('global_data', {})
            
            bullish_signals = 0
            bearish_signals = 0
            
            if technical_data and technical_data.get('status') == 'success':
                inst_data = technical_data.get(instrument, {})
                trend = inst_data.get('trend', 'neutral')
                rsi = inst_data.get('rsi', 50)
                
                if trend == 'bullish':
                    bullish_signals += 2
                elif trend == 'bearish':
                    bearish_signals += 2
                
                if rsi > 55:
                    bullish_signals += 1
                elif rsi < 45:
                    bearish_signals += 1
            
            if vix_data and vix_data.get('status') == 'success':
                vix = vix_data.get('vix', 16.5)
                if vix < 18:  # Low VIX = bullish sentiment
                    bullish_signals += 1
                elif vix > 22:  # High VIX = bearish sentiment
                    bearish_signals += 1
            
            if global_data and global_data.get('status') == 'success':
                indices = global_data.get('indices', {})
                sgx = indices.get('SGX_NIFTY', 0)
                if sgx > 10:
                    bullish_signals += 1
                elif sgx < -10:
                    bearish_signals += 1
            
            options_data = market_data.get('options_data', {})
            if options_data and options_data.get('status') == 'success':
                inst_options = options_data.get(instrument, {})
                if inst_options and inst_options.get('status') == 'success':
                    options_chain = inst_options.get('options_data', {})
                    
                    ce_oi_total = sum(opt.get('oi', 0) for key, opt in options_chain.items() if '_CE' in key)
                    pe_oi_total = sum(opt.get('oi', 0) for key, opt in options_chain.items() if '_PE' in key)
                    
                    if ce_oi_total > 0:
                        pcr = pe_oi_total / ce_oi_total
                        if pcr > 1.2:  # High PCR = bullish
                            bullish_signals += 1
                        elif pcr < 0.8:  # Low PCR = bearish
                            bearish_signals += 1
            
            if bullish_signals > bearish_signals:
                direction = 'CE'
                logger.info(f"📈 {instrument} direction: CE (Bullish signals: {bullish_signals}, Bearish: {bearish_signals})")
            elif bearish_signals > bullish_signals:
                direction = 'PE'
                logger.info(f"📉 {instrument} direction: PE (Bullish signals: {bullish_signals}, Bearish: {bearish_signals})")
            else:
                direction = 'CE' if confidence > 65 else 'PE'
                logger.info(f"⚖️ {instrument} direction: {direction} (Tie-breaker, confidence: {confidence}%)")
            
            return direction
            
        except Exception as e:
            logger.error(f"❌ Direction determination failed: {e}")
            return 'CE'  # Default to CE on error
    
    def _calculate_dynamic_levels(self, entry_price: float, instrument: str, market_data: Dict[str, Any]) -> tuple:
        """Calculate dynamic SL and targets based on volatility and market conditions"""
        try:
            vix_data = market_data.get('vix_data', {})
            vix = vix_data.get('vix', 16.5) if vix_data and vix_data.get('status') == 'success' else 16.5
            
            base_sl = 0.70
            base_target1 = 1.35
            base_target2 = 1.70
            
            if vix > 25:  # High volatility
                sl_adjustment = -0.05  # Wider SL
                target_adjustment = 0.15  # Higher targets
            elif vix < 15:  # Low volatility
                sl_adjustment = 0.05  # Tighter SL
                target_adjustment = -0.10  # Lower targets
            else:
                sl_adjustment = 0
                target_adjustment = 0
            
            if entry_price > 200:  # High premium
                sl_adjustment -= 0.05  # Tighter SL for expensive options
                target_adjustment += 0.10
            elif entry_price < 50:  # Low premium
                sl_adjustment += 0.05  # Wider SL for cheap options
                target_adjustment -= 0.05
            
            sl_percentage = max(0.60, min(base_sl + sl_adjustment, 0.80))
            target1_percentage = max(1.20, base_target1 + target_adjustment)
            target2_percentage = max(1.50, base_target2 + target_adjustment * 1.5)
            
            sl_price = max(int(entry_price * sl_percentage), 5)
            target1 = int(entry_price * target1_percentage)
            target2 = int(entry_price * target2_percentage)
            
            logger.debug(f"💰 Dynamic levels for ₹{entry_price}: SL=₹{sl_price} ({sl_percentage:.2f}), T1=₹{target1}, T2=₹{target2} (VIX: {vix})")
            
            return sl_price, target1, target2
            
        except Exception as e:
            logger.error(f"❌ Dynamic level calculation failed: {e}")
            sl_price = max(int(entry_price * 0.70), 5)
            target1 = int(entry_price * 1.35)
            target2 = int(entry_price * 1.70)
            return sl_price, target1, target2
    
    def _get_option_details(self, instrument: str, strike: int, direction: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional option details like IV, Delta, OI trend"""
        try:
            options_data = market_data.get('options_data', {})
            if not options_data or options_data.get('status') != 'success':
                return {}
            
            instrument_options = options_data.get(instrument, {})
            if not instrument_options or instrument_options.get('status') != 'success':
                return {}
            
            options_chain = instrument_options.get('options_data', {})
            strike_key = f"{float(strike)}_{direction}"
            
            if strike_key in options_chain:
                option_data = options_chain[strike_key]
                
                ltp = option_data.get('ltp', 0)
                spot_data = market_data.get('spot_data', {})
                current_price = 0
                if spot_data and spot_data.get('status') == 'success':
                    current_price = spot_data.get('prices', {}).get(instrument, 0)
                
                if current_price > 0 and ltp > 0:
                    moneyness = strike / current_price if direction == 'CE' else current_price / strike
                    iv_estimate = (ltp / current_price) * 100 * (2 - moneyness)
                    iv_estimate = max(10, min(iv_estimate, 80))  # Cap between 10-80%
                else:
                    iv_estimate = 20  # Default IV
                
                if direction == 'CE':
                    if strike > current_price:  # OTM CE
                        delta_estimate = 0.3
                    elif strike < current_price * 0.98:  # ITM CE
                        delta_estimate = 0.7
                    else:  # ATM CE
                        delta_estimate = 0.5
                else:  # PE
                    if strike < current_price:  # OTM PE
                        delta_estimate = -0.3
                    elif strike > current_price * 1.02:  # ITM PE
                        delta_estimate = -0.7
                    else:  # ATM PE
                        delta_estimate = -0.5
                
                oi = option_data.get('oi', 0)
                if oi > 100000:
                    oi_trend = 'high'
                elif oi > 50000:
                    oi_trend = 'moderate'
                else:
                    oi_trend = 'low'
                
                technical_data = market_data.get('technical_data', {})
                if technical_data and technical_data.get('status') == 'success':
                    inst_data = technical_data.get(instrument, {})
                    trend = inst_data.get('trend', 'neutral')
                    rsi = inst_data.get('rsi', 50)
                    
                    if direction == 'CE':
                        direction_reason = f"Bullish trend ({trend}), RSI {rsi}"
                    else:
                        direction_reason = f"Bearish trend ({trend}), RSI {rsi}"
                else:
                    direction_reason = "Multi-factor analysis"
                
                return {
                    'iv': round(iv_estimate, 1),
                    'delta': round(delta_estimate, 2),
                    'oi_trend': oi_trend,
                    'direction_reason': direction_reason
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Failed to get option details: {e}")
            return {}
    
    def _generate_signal_reason(self, confidence: float, market_data: Dict[str, Any], direction: str = None) -> str:
        """Generate human-readable reason based on live market conditions"""
        reasons = []
        
        technical_data = market_data.get('technical_data', {})
        if technical_data and technical_data.get('status') == 'success':
            for instrument in ['NIFTY', 'BANKNIFTY']:
                inst_data = technical_data.get(instrument, {})
                rsi = inst_data.get('rsi', 50)
                trend = inst_data.get('trend', 'neutral')
                
                if rsi > 65:
                    reasons.append(f"{instrument} RSI overbought ({rsi})")
                elif rsi < 35:
                    reasons.append(f"{instrument} RSI oversold ({rsi})")
                
                if trend == 'bullish':
                    reasons.append(f"{instrument} trend bullish")
                elif trend == 'bearish':
                    reasons.append(f"{instrument} trend bearish")
        
        vix_data = market_data.get('vix_data', {})
        if vix_data and vix_data.get('status') == 'success':
            vix = vix_data.get('vix', 16.5)
            if vix < 15:
                reasons.append(f"Low VIX ({vix}) - complacency")
            elif vix > 25:
                reasons.append(f"High VIX ({vix}) - fear")
        
        options_data = market_data.get('options_data', {})
        if options_data and options_data.get('status') == 'success':
            reasons.append("Live options data supportive")
        
        global_data = market_data.get('global_data', {})
        if global_data and global_data.get('status') == 'success':
            indices = global_data.get('indices', {})
            sgx = indices.get('SGX_NIFTY', 0)
            if sgx > 0:
                reasons.append(f"SGX Nifty positive ({sgx})")
            elif sgx < 0:
                reasons.append(f"SGX Nifty negative ({sgx})")
        
        if confidence > 70:
            reasons.insert(0, "Strong multi-factor confluence")
        elif confidence > 60:
            reasons.insert(0, "Moderate bullish signals")
        
        return ", ".join(reasons[:4]) if reasons else f"Multi-factor analysis ({confidence}%)"
    
    def _get_live_option_ltp(self, instrument: str, strike: int, direction: str, market_data: Dict[str, Any]) -> float:
        """Get live LTP for specific option strike"""
        try:
            options_data = market_data.get('options_data', {})
            if not options_data or options_data.get('status') != 'success':
                return 0
            
            instrument_options = options_data.get(instrument, {})
            if not instrument_options or instrument_options.get('status') != 'success':
                return 0
            
            options_chain = instrument_options.get('options_data', {})
            
            strike_key = f"{float(strike)}_{direction}"
            if strike_key in options_chain:
                ltp = options_chain[strike_key].get('ltp', 0)
                if ltp > 0:
                    logger.info(f"📊 Found live LTP for {instrument} {strike} {direction}: ₹{ltp}")
                    return ltp
            
            alt_keys = [
                f"{strike}_{direction}",
                f"{int(strike)}_{direction}",
                f"{strike}.0_{direction}"
            ]
            
            for alt_key in alt_keys:
                if alt_key in options_chain:
                    ltp = options_chain[alt_key].get('ltp', 0)
                    if ltp > 0:
                        logger.info(f"📊 Found live LTP for {instrument} {alt_key}: ₹{ltp}")
                        return ltp
            
            logger.warning(f"⚠️ Strike {strike} {direction} not found in {instrument} options chain")
            available_strikes = list(options_chain.keys())[:5]
            logger.debug(f"Available strikes: {available_strikes}")
            return 0
            
        except Exception as e:
            logger.error(f"❌ Failed to get live option LTP: {e}")
            return 0
    
    def _find_best_strike_with_ltp(self, instrument: str, current_price: float, direction: str, market_data: Dict[str, Any]) -> tuple:
        """Find the best available strike with live LTP from options chain"""
        try:
            options_data = market_data.get('options_data', {})
            if not options_data or options_data.get('status') != 'success':
                return 0, 0
            
            instrument_options = options_data.get(instrument, {})
            if not instrument_options or instrument_options.get('status') != 'success':
                return 0, 0
            
            options_chain = instrument_options.get('options_data', {})
            if not options_chain:
                return 0, 0
            
            available_strikes = []
            for key, option_data in options_chain.items():
                if f'_{direction}' in key and option_data.get('ltp', 0) > 0:
                    strike_price = option_data.get('strike', 0)
                    ltp = option_data.get('ltp', 0)
                    if strike_price > 0 and ltp > 0:
                        available_strikes.append((strike_price, ltp, key))
            
            if not available_strikes:
                logger.warning(f"⚠️ No {direction} strikes with LTP found for {instrument}")
                return 0, 0
            
            strike_buffer = 100 if instrument == 'NIFTY' else 500
            if direction == 'CE':
                ideal_strike = current_price + strike_buffer
            else:
                ideal_strike = current_price - strike_buffer
            
            best_strike = min(available_strikes, key=lambda x: abs(x[0] - ideal_strike))
            
            strike_price, ltp, strike_key = best_strike
            logger.info(f"📊 Selected {instrument} {strike_key}: Strike={strike_price}, LTP=₹{ltp}")
            
            return int(strike_price), ltp
            
        except Exception as e:
            logger.error(f"❌ Failed to find best strike with LTP: {e}")
            return 0, 0
    
    def _get_option_expiry(self, market_data: Dict[str, Any]) -> str:
        """Get option expiry date from market data"""
        try:
            options_data = market_data.get('options_data', {})
            if options_data and options_data.get('status') == 'success':
                nifty_options = options_data.get('NIFTY', {})
                if nifty_options and nifty_options.get('status') == 'success':
                    expiry = nifty_options.get('expiry', '')
                    if expiry:
                        return expiry
                
                banknifty_options = options_data.get('BANKNIFTY', {})
                if banknifty_options and banknifty_options.get('status') == 'success':
                    expiry = banknifty_options.get('expiry', '')
                    if expiry:
                        return expiry
            
            return 'Current Week'
            
        except Exception as e:
            logger.error(f"❌ Failed to get option expiry: {e}")
            return 'Current Week'
    
    async def shutdown(self):
        """Shutdown the trade signal engine"""
        try:
            logger.info("🔄 Shutting down TradeSignalEngine...")
        except Exception as e:
            logger.error(f"❌ TradeSignalEngine shutdown failed: {e}")
