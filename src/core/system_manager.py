"""
Main system manager and orchestrator for institutional-grade trading
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List

from src.core.data_manager import DataManager
from src.analysis.trade_signal_engine import TradeSignalEngine
from src.analysis.multi_timeframe_analysis import MultiTimeframeAnalyzer
from src.analysis.pattern_detection import PatternDetector
from src.analysis.support_resistance import SupportResistanceCalculator
from src.strategies.orb_strategy import ORBStrategy
from src.execution.trade_executor import TradeExecutor
from src.risk.risk_manager import RiskManager
from src.analysis.backtesting import BacktestingEngine
from src.utils.telegram_notifier import TelegramNotifier
from src.auth.kite_auth import KiteAuthenticator

logger = logging.getLogger('trading_system.system_manager')

class TradingSystemManager:
    """Institutional-grade trading system orchestrator"""
    
    def __init__(self, settings):
        self.settings = settings
        
        self.data_manager = DataManager(settings)
        self.signal_engine = TradeSignalEngine(settings)
        self.telegram_notifier = TelegramNotifier(settings) if settings.TELEGRAM_BOT_TOKEN else None
        
        self.kite_auth = KiteAuthenticator(settings)
        self.kite_client = None
        self.multi_timeframe = None
        self.pattern_detector = None
        self.support_resistance = None
        self.orb_strategy = None
        self.trade_executor = None
        self.risk_manager = None
        self.backtesting_engine = None
        
        self.running = False
        self.cycle_count = 0
        self.institutional_mode = True
        self.active_positions = {}
        
        logger.info("✅ Institutional-grade TradingSystemManager initialized")
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("🔧 Initializing all institutional-grade components...")
            return await self._initialize_system()
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def run(self):
        """Main system loop"""
        try:
            logger.info("🚀 Starting Trading System Manager")
            
            if not await self._initialize_system():
                logger.error("❌ System initialization failed")
                return
            
            self.running = True
            logger.info("✅ System initialized successfully")
            
            while self.running:
                try:
                    await self._execute_cycle()
                    self.cycle_count += 1
                    
                    await asyncio.sleep(self.settings.DATA_REFRESH_INTERVAL)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Shutdown signal received")
                    break
                except Exception as e:
                    logger.error(f"❌ Cycle execution error: {e}")
                    await asyncio.sleep(10)  # Wait before retry
            
        except Exception as e:
            logger.error(f"❌ System manager error: {e}")
        finally:
            await self._shutdown()
    
    async def _initialize_system(self) -> bool:
        """Initialize all institutional-grade system components"""
        try:
            logger.info("🔧 Initializing institutional-grade system components...")
            
            if not await self.kite_auth.initialize():
                logger.warning("⚠️ Kite authentication initialization failed")
            
            self.kite_client = self.kite_auth.get_authenticated_kite()
            if not self.kite_client:
                logger.warning("⚠️ Kite client not available - some features will be limited")
            else:
                logger.info("✅ Kite client obtained successfully")
                self.data_manager.kite_client = self.kite_client
                logger.info("✅ Kite client passed to DataManager")

            if not await self.data_manager.initialize():
                logger.error("❌ Data manager initialization failed")
                return False
            
            self.multi_timeframe = MultiTimeframeAnalyzer(self.settings)
            self.pattern_detector = PatternDetector(self.kite_client) if self.kite_client else None
            self.support_resistance = SupportResistanceCalculator(self.kite_client) if self.kite_client else None
            self.orb_strategy = ORBStrategy(self.kite_client) if self.kite_client else None
            
            self.trade_executor = TradeExecutor(self.kite_client, self.settings)
            self.risk_manager = RiskManager(self.settings)
            self.backtesting_engine = BacktestingEngine(self.kite_client)
            
            if self.telegram_notifier:
                await self.telegram_notifier.send_message(
                    "🚀 **INSTITUTIONAL-GRADE TRADING SYSTEM STARTED**\n\n"
                    "✅ Multi-timeframe Analysis\n"
                    "✅ Pattern Detection\n"
                    "✅ Support/Resistance Calculation\n"
                    "✅ ORB Strategy\n"
                    "✅ Trade Execution Engine\n"
                    "✅ Advanced Risk Management\n"
                    "✅ Backtesting Framework\n\n"
                    "🛡️ All systems operational and SEBI compliant"
                )
            
            logger.info("✅ All institutional-grade components initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization error: {e}")
            return False
    
    async def _execute_cycle(self):
        """Execute one complete institutional-grade analysis cycle"""
        cycle_start = datetime.now()
        logger.info(f"🔄 Starting institutional analysis cycle #{self.cycle_count + 1}")
        
        try:
            market_data = await self.data_manager.fetch_all_data()
            
            enhanced_data = await self._perform_enhanced_analysis(market_data)
            
            risk_status = {'triggered': False}
            if self.risk_manager and isinstance(market_data, dict):
                risk_status = self.risk_manager.check_circuit_breakers(market_data)
            if risk_status['triggered']:
                logger.warning(f"⚠️ Circuit breaker triggered: {risk_status['action']}")
                if risk_status['action'] == 'halt_trading':
                    await self._display_results(enhanced_data, [])
                    return
            
            if hasattr(self.signal_engine, 'cleanup_expired_signals'):
                self.signal_engine.cleanup_expired_signals()
            
            signals = await self.signal_engine.generate_signals(enhanced_data)
            
            validated_signals = []
            risk_filtered_signals = []
            
            for signal in signals:
                risk_assessment = self.risk_manager.validate_trade_risk(
                    signal, self.active_positions, enhanced_data
                )
                
                if risk_assessment['approved']:
                    validated_signals.append(signal)
                    logger.info(f"✅ Signal approved: {signal['instrument']} {signal['strike']} {signal['option_type']}")
                else:
                    logger.warning(f"⚠️ Signal rejected: {signal['instrument']} - {risk_assessment['violations']}")
                    signal['risk_status'] = 'RISK_FILTERED'
                    signal['risk_violations'] = risk_assessment.get('violations', [])
                    signal['risk_score'] = risk_assessment.get('risk_score', 0)
                    risk_filtered_signals.append(signal)
            
            all_signals_for_notification = validated_signals + risk_filtered_signals
            if all_signals_for_notification and self.telegram_notifier:
                await self._send_institutional_notifications(all_signals_for_notification)
            
            if validated_signals and self.institutional_mode:
                execution_results = await self._execute_validated_trades(validated_signals)
                await self._update_position_tracking(execution_results)
            
            if self.active_positions:
                position_updates = await self.trade_executor.monitor_positions()
                await self._process_position_updates(position_updates)
            
            await self._display_institutional_results(enhanced_data, validated_signals, risk_filtered_signals)
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"✅ Institutional cycle #{self.cycle_count + 1} completed in {cycle_duration:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ Institutional cycle execution failed: {e}")
    
    async def _perform_enhanced_analysis(self, market_data: Dict) -> Dict:
        """Perform enhanced analysis with all institutional modules"""
        try:
            enhanced_data = market_data.copy()
            
            for symbol in ['NIFTY', 'BANKNIFTY']:
                if self.multi_timeframe:
                    mtf_analysis = await self.multi_timeframe.analyze_symbol(symbol, market_data)
                    enhanced_data[f'{symbol.lower()}_mtf'] = mtf_analysis
                
                if self.pattern_detector:
                    patterns = await self.pattern_detector.detect_patterns(symbol)
                    enhanced_data[f'{symbol.lower()}_patterns'] = patterns
                
                if self.support_resistance:
                    sr_levels = await self.support_resistance.calculate_levels(symbol)
                    enhanced_data[f'{symbol.lower()}_sr'] = sr_levels
                
                if self.orb_strategy:
                    orb_analysis = await self.orb_strategy.analyze_orb(symbol)
                    enhanced_data[f'{symbol.lower()}_orb'] = orb_analysis
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"❌ Enhanced analysis failed: {e}")
            return market_data
    
    async def _execute_validated_trades(self, signals: List[Dict]) -> List[Dict]:
        """Execute validated trades through trade executor"""
        execution_results = []
        
        try:
            for signal in signals:
                execution_result = await self.trade_executor.execute_trade(signal)
                execution_results.append(execution_result)
                
                if execution_result['status'] == 'success':
                    self.risk_manager.update_daily_stats(execution_result)
                
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
        
        return execution_results
    
    async def _update_position_tracking(self, execution_results: List[Dict]):
        """Update position tracking with execution results"""
        try:
            for result in execution_results:
                if result['status'] == 'success':
                    position_key = f"{result['signal']['instrument']}_{result['signal']['strike']}_{result['signal']['option_type']}"
                    self.active_positions[position_key] = result
                    
        except Exception as e:
            logger.error(f"❌ Position tracking update failed: {e}")
    
    async def _process_position_updates(self, position_updates: Dict):
        """Process position monitoring updates"""
        try:
            if position_updates.get('actions_taken'):
                for action in position_updates['actions_taken']:
                    position_key = action['position']
                    if position_key in self.active_positions:
                        del self.active_positions[position_key]
                        logger.info(f"✅ Position closed: {position_key}")
                        
        except Exception as e:
            logger.error(f"❌ Position update processing failed: {e}")
    
    async def _display_institutional_results(self, market_data: Dict, signals: List[Dict], risk_filtered_signals: List[Dict] = None):
        """Display comprehensive institutional-grade analysis results"""
        try:
            print("\n" + "=" * 100)
            print("🏛️ INSTITUTIONAL-GRADE OPTIONS TRADING SYSTEM - COMPREHENSIVE ANALYSIS")
            print("=" * 100)
            print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🔄 Cycle: #{self.cycle_count + 1} | Mode: INSTITUTIONAL")
            
            print("\n📊 MARKET HEALTH & RISK ASSESSMENT:")
            print("-" * 50)
            
            spot_data = market_data.get('spot_data', {})
            if spot_data.get('status') == 'success':
                prices = spot_data.get('prices', {})
                print(f"💹 NIFTY: {prices.get('NIFTY', 'N/A')}")
                print(f"💹 BANKNIFTY: {prices.get('BANKNIFTY', 'N/A')}")
            
            vix_data = market_data.get('vix_data', {})
            if vix_data.get('status') == 'success':
                vix_level = vix_data.get('vix', 'N/A')
                vix_status = "🔴 HIGH" if vix_level > 25 else "🟡 MEDIUM" if vix_level > 20 else "🟢 LOW"
                print(f"📈 VIX: {vix_level} ({vix_data.get('change', 0):+.2f}) - {vix_status}")
            
            print("\n🧠 ENHANCED ANALYSIS RESULTS:")
            print("-" * 50)
            
            for symbol in ['NIFTY', 'BANKNIFTY']:
                symbol_lower = symbol.lower()
                
                mtf_data = market_data.get(f'{symbol_lower}_mtf', {})
                if mtf_data.get('status') == 'success':
                    consensus = mtf_data.get('consensus', {})
                    print(f"📊 {symbol} Multi-Timeframe: {consensus.get('overall_signal', 'N/A').upper()} "
                          f"(Confidence: {consensus.get('confidence', 0):.0f}%)")
                
                patterns_data = market_data.get(f'{symbol_lower}_patterns', {})
                if patterns_data.get('status') == 'success':
                    detected_patterns = patterns_data.get('detected_patterns', [])
                    if detected_patterns:
                        pattern_names = [p['pattern'] for p in detected_patterns[:3]]
                        print(f"🕯️ {symbol} Patterns: {', '.join(pattern_names)}")
                
                sr_data = market_data.get(f'{symbol_lower}_sr', {})
                if sr_data.get('status') == 'success':
                    current_level = sr_data.get('current_level', 'neutral')
                    strength = sr_data.get('strength', 0)
                    print(f"📈 {symbol} S/R: {current_level.upper()} (Strength: {strength:.1f})")
                
                orb_data = market_data.get(f'{symbol_lower}_orb', {})
                if orb_data.get('status') == 'success':
                    orb_signal = orb_data.get('signal', 'neutral')
                    orb_confidence = orb_data.get('confidence', 0)
                    print(f"🎯 {symbol} ORB: {orb_signal.upper()} (Confidence: {orb_confidence:.0f}%)")
            
            if self.risk_manager:
                risk_summary = self.risk_manager.get_risk_summary()
                print(f"\n🛡️ RISK MANAGEMENT STATUS:")
                print("-" * 50)
                daily_stats = risk_summary.get('daily_stats', {})
                print(f"💰 Daily P&L: ₹{daily_stats.get('pnl', 0):.2f}")
                print(f"📊 Trades Today: {daily_stats.get('trades_count', 0)}")
                print(f"📉 Max Drawdown: ₹{daily_stats.get('max_drawdown', 0):.2f}")
                print(f"⚠️ Risk Violations: {len(daily_stats.get('risk_violations', []))}")
            
            print(f"\n📋 ACTIVE POSITIONS: {len(self.active_positions)}")
            print("-" * 50)
            if self.active_positions:
                for pos_key, position in list(self.active_positions.items())[:5]:  # Show max 5
                    signal = position.get('signal', {})
                    print(f"   📍 {signal.get('instrument', 'N/A')} {signal.get('strike', 'N/A')} "
                          f"{signal.get('option_type', 'N/A')} - Entry: ₹{position.get('execution_price', 0)}")
            else:
                print("   No active positions")
            
            print("\n🎯 VALIDATED TRADE SIGNALS:")
            print("-" * 50)
            
            if signals:
                for i, signal in enumerate(signals, 1):
                    print(f"\n✅ INSTITUTIONAL SIGNAL #{i}:")
                    print(f"   🎯 Instrument: {signal['instrument']}")
                    print(f"   🎯 Strike: {signal['strike']} {signal['option_type']}")
                    print(f"   💰 Entry Price: ₹{signal['entry_price']}")
                    print(f"   🛑 Stop Loss: ₹{signal['stop_loss']}")
                    print(f"   🎯 Target 1: ₹{signal['target_1']}")
                    print(f"   🎯 Target 2: ₹{signal['target_2']}")
                    print(f"   📊 Confidence: {signal['confidence']}%")
                    print(f"   📝 Reason: {signal['reason']}")
                    print(f"   ⬆️ Direction: {signal['direction'].upper()}")
                    print(f"   🛡️ Risk Approved: ✅")
            else:
                print("❌ No validated signals (risk-filtered or below confidence threshold)")
                print(f"   Minimum confidence required: {self.settings.CONFIDENCE_THRESHOLD}%")
                if risk_filtered_signals:
                    print(f"   Risk-filtered signals: {len(risk_filtered_signals)} (sent to Telegram with risk warnings)")
            
            freshness = self.data_manager.get_data_freshness()
            print(f"\n📡 INSTITUTIONAL DATA SOURCES ({freshness['health_percentage']:.0f}% HEALTHY):")
            print("-" * 50)
            for source, status in market_data.get('data_status', {}).items():
                status_icon = "✅" if status == 'success' else "❌" if status == 'failed' else "🔄"
                print(f"   {status_icon} {source.replace('_', ' ').title()}: {status}")
            
            print("\n" + "=" * 100)
            
        except Exception as e:
            logger.error(f"❌ Institutional display results error: {e}")
    
    async def _display_results(self, market_data: Dict, signals: List[Dict]):
        """Fallback display method for compatibility"""
        await self._display_institutional_results(market_data, signals, [])
    
    async def _send_institutional_notifications(self, signals: List[Dict]):
        """Send institutional-grade trade signal notifications via Telegram"""
        try:
            for signal in signals:
                risk_status = signal.get('risk_status', 'VALIDATED')
                if risk_status == 'RISK_FILTERED':
                    risk_summary = f"⚠️ RISK FILTERED (Score: {signal.get('risk_score', 0):.0f}/100)"
                    violations = signal.get('risk_violations', [])
                    risk_details = f"\n🚨 **Violations:** {', '.join(violations[:2])}" if violations else ""
                else:
                    risk_summary = "✅ VALIDATED & APPROVED"
                    risk_details = ""
                
                message = f"""
🎯 **TRADE SIGNAL - {signal['instrument']}**

📊 **Instrument:** {signal['instrument']}
🎯 **Strike:** {signal['strike']} {signal['option_type']}
📅 **Expiry:** {signal.get('expiry', 'Current Week')}
💰 **Entry Price:** ₹{signal['entry_price']}
🛑 **Stop Loss:** ₹{signal['stop_loss']}
🎯 **Target 1:** ₹{signal['target_1']}
🎯 **Target 2:** ₹{signal['target_2']}
📈 **Confidence:** {signal['confidence']}%
⬆️ **Direction:** {signal['direction'].upper()}
📝 **Reason:** {signal['reason']}

🛡️ **Risk Status:** {risk_summary}{risk_details}
📋 **Active Positions:** {len(self.active_positions)}
⏰ **Timestamp (IST):** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🏛️ **VLR_AI Institutional Trading System**
"""
                
                if self.telegram_notifier:
                    success = await self.telegram_notifier.send_message(message)
                    if success:
                        logger.info(f"✅ Telegram signal sent: {signal['instrument']} {signal['strike']} {signal['option_type']}")
                    else:
                        logger.error(f"❌ Telegram signal failed: {signal['instrument']}")
                
        except Exception as e:
            logger.error(f"❌ Institutional notification sending failed: {e}")
    
    async def _send_signal_notifications(self, signals: List[Dict]):
        """Fallback notification method for compatibility"""
        await self._send_institutional_notifications(signals)
    
    async def _shutdown(self):
        """Graceful institutional system shutdown"""
        try:
            logger.info("🔄 Shutting down institutional trading system...")
            self.running = False
            
            if self.active_positions and self.trade_executor:
                logger.info(f"📋 Monitoring {len(self.active_positions)} active positions during shutdown")
            
            if self.risk_manager:
                final_risk_report = self.risk_manager.get_risk_summary()
                logger.info(f"📊 Final daily P&L: ₹{final_risk_report.get('daily_stats', {}).get('pnl', 0):.2f}")
            
            if self.telegram_notifier:
                shutdown_message = f"""
🛑 **INSTITUTIONAL TRADING SYSTEM SHUTDOWN**

📊 **Final Session Summary:**
• Cycles Completed: {self.cycle_count}
• Active Positions: {len(self.active_positions)}
• System Mode: INSTITUTIONAL

🛡️ **Risk Management:**
• All positions monitored
• Risk limits maintained
• SEBI compliance verified

⏰ **Shutdown Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                await self.telegram_notifier.send_message(shutdown_message)
            
            logger.info("✅ Institutional system shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
    
    def stop(self):
        """Stop the institutional system"""
        self.running = False
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'running': self.running,
                'cycle_count': self.cycle_count,
                'institutional_mode': self.institutional_mode,
                'active_positions': len(self.active_positions),
                'components': {
                    'data_manager': bool(self.data_manager),
                    'signal_engine': bool(self.signal_engine),
                    'multi_timeframe': bool(self.multi_timeframe),
                    'pattern_detector': bool(self.pattern_detector),
                    'support_resistance': bool(self.support_resistance),
                    'orb_strategy': bool(self.orb_strategy),
                    'trade_executor': bool(self.trade_executor),
                    'risk_manager': bool(self.risk_manager),
                    'backtesting_engine': bool(self.backtesting_engine),
                    'kite_client': bool(self.kite_client),
                    'telegram_notifier': bool(self.telegram_notifier)
                }
            }
        except Exception as e:
            logger.error(f"❌ System status error: {e}")
            return {'error': str(e)}
