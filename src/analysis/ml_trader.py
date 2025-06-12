"""
TradeMind_AI: Self-Learning AI Module
Tracks patterns, learns from trades, and improves strategies
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class SelfLearningTrader:
    def __init__(self):
        """Initialize Self-Learning AI Module"""
        print("🧠 Initializing Self-Learning AI Module...")
        
        # Model storage
        self.model_dir = "models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # Learning parameters
        self.min_trades_for_learning = 50
        self.confidence_threshold = 0.70
        self.feature_importance_threshold = 0.05
        
        # Models for different strategies
        self.models = {
            'trend_predictor': None,
            'strike_selector': None,
            'risk_manager': None,
            'confidence_scorer': None
        }
        
        # Performance tracking
        self.performance_history = []
        self.winning_patterns = []
        self.losing_patterns = []
        
        # Load existing models if available
        self.load_models()
        
        print("✅ Self-Learning AI ready!")
    
    def extract_trade_features(self, trade_data):
        """Extract features from trade data for learning"""
        features = {
            # Market conditions
            'rsi': trade_data.get('rsi', 50),
            'macd_signal': trade_data.get('macd_signal', 0),
            'bollinger_position': trade_data.get('bollinger_position', 0.5),
            'volume_ratio': trade_data.get('volume_ratio', 1.0),
            'oi_ratio': trade_data.get('oi_ratio', 1.0),
            'iv_skew': trade_data.get('iv_skew', 0),
            
            # Trade specifics
            'moneyness': trade_data.get('moneyness', 1.0),
            'days_to_expiry': trade_data.get('days_to_expiry', 7),
            'time_of_day': self._get_time_score(trade_data.get('timestamp')),
            'day_of_week': self._get_day_score(trade_data.get('timestamp')),
            
            # Market sentiment
            'vix_level': trade_data.get('vix_level', 15),
            'put_call_ratio': trade_data.get('pcr', 1.0),
            'market_trend': trade_data.get('trend_score', 0),
            
            # Greeks
            'delta': abs(trade_data.get('delta', 0.3)),
            'gamma': trade_data.get('gamma', 0.01),
            'theta': trade_data.get('theta', -5),
            'vega': trade_data.get('vega', 10),
            
            # Previous performance
            'last_3_trades_winrate': self._get_recent_performance(3),
            'last_10_trades_winrate': self._get_recent_performance(10),
            'confidence_score': trade_data.get('ai_confidence', 80) / 100
        }
        
        return features
    
    def _get_time_score(self, timestamp):
        """Convert time to numerical score (0-1)"""
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        hour = timestamp.hour
        minute = timestamp.minute
        
        # Market hours: 9:15 AM to 3:30 PM
        minutes_since_open = (hour - 9) * 60 + minute - 15
        total_market_minutes = 375  # 6.25 hours
        
        return min(max(minutes_since_open / total_market_minutes, 0), 1)
    
    def _get_day_score(self, timestamp):
        """Convert day of week to score"""
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return timestamp.weekday() / 4  # 0-1 scale (Mon=0, Fri=1)
    
    def _get_recent_performance(self, n_trades):
        """Get win rate of last n trades"""
        if len(self.performance_history) < n_trades:
            return 0.5  # Default 50%
        
        recent = self.performance_history[-n_trades:]
        wins = sum(1 for t in recent if t['profit'] > 0)
        return wins / n_trades
    
    def learn_from_trade(self, trade_result):
        """Learn from completed trade"""
        try:
            # Extract features
            features = self.extract_trade_features(trade_result)
            
            # Determine outcome
            profit = trade_result.get('pnl', 0)
            outcome = 1 if profit > 0 else 0
            
            # Store in performance history
            self.performance_history.append({
                'features': features,
                'outcome': outcome,
                'profit': profit,
                'timestamp': datetime.now()
            })
            
            # Identify patterns
            if outcome == 1:
                self.winning_patterns.append(features)
            else:
                self.losing_patterns.append(features)
            
            # Retrain models if enough data
            if len(self.performance_history) >= self.min_trades_for_learning:
                self.train_models()
            
            # Save updated state
            self.save_learning_state()
            
            print(f"📚 Learned from trade: {'WIN' if outcome else 'LOSS'} (₹{profit:.2f})")
            
        except Exception as e:
            print(f"❌ Learning error: {e}")
    
    def train_models(self):
        """Train all ML models"""
        print("\n🎯 Training AI models with accumulated data...")
        
        # Prepare data
        X, y = self.prepare_training_data()
        
        if len(X) < self.min_trades_for_learning:
            print("⚠️ Not enough data for training")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models_config = {
            'trend_predictor': RandomForestClassifier(n_estimators=100, random_state=42),
            'strike_selector': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'risk_manager': RandomForestClassifier(n_estimators=50, random_state=42),
            'confidence_scorer': GradientBoostingClassifier(n_estimators=50, random_state=42)
        }
        
        for model_name, model in models_config.items():
            print(f"\n📊 Training {model_name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ {model_name} Accuracy: {accuracy:.2%}")
            
            # Store model
            self.models[model_name] = {
                'model': model,
                'scaler': scaler,
                'accuracy': accuracy,
                'feature_importance': self._get_feature_importance(model, X.columns)
            }
        
        # Save models
        self.save_models()
        
        # Generate insights
        self.generate_learning_insights()
    
    def prepare_training_data(self):
        """Prepare data for model training"""
        if not self.performance_history:
            return pd.DataFrame(), pd.Series()
        
        # Convert to DataFrame
        features_list = []
        outcomes = []
        
        for trade in self.performance_history:
            features_list.append(trade['features'])
            outcomes.append(trade['outcome'])
        
        X = pd.DataFrame(features_list)
        y = pd.Series(outcomes)
        
        return X, y
    
    def _get_feature_importance(self, model, feature_names):
        """Get feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_imp[feature_imp['importance'] > self.feature_importance_threshold]
        
        return pd.DataFrame()
    
    def predict_trade_outcome(self, trade_features):
        """Predict trade outcome using trained models"""
        if not self.models['trend_predictor']:
            return {
                'prediction': 0.5,
                'confidence': 50,
                'recommendation': 'NEUTRAL',
                'insights': ['No trained model available']
            }
        
        try:
            # Prepare features
            features_df = pd.DataFrame([trade_features])
            
            # Get predictions from all models
            predictions = {}
            confidences = {}
            
            for model_name, model_data in self.models.items():
                if model_data:
                    scaler = model_data['scaler']
                    model = model_data['model']
                    
                    # Scale features
                    features_scaled = scaler.transform(features_df)
                    
                    # Predict
                    pred_proba = model.predict_proba(features_scaled)[0]
                    predictions[model_name] = pred_proba[1]  # Probability of success
                    confidences[model_name] = max(pred_proba) * 100
            
            # Ensemble prediction
            ensemble_prediction = np.mean(list(predictions.values()))
            ensemble_confidence = np.mean(list(confidences.values()))
            
            # Generate recommendation
            if ensemble_prediction > 0.7:
                recommendation = 'STRONG BUY'
            elif ensemble_prediction > 0.6:
                recommendation = 'BUY'
            elif ensemble_prediction < 0.3:
                recommendation = 'AVOID'
            elif ensemble_prediction < 0.4:
                recommendation = 'CAUTION'
            else:
                recommendation = 'NEUTRAL'
            
            # Generate insights
            insights = self._generate_prediction_insights(trade_features, predictions)
            
            return {
                'prediction': ensemble_prediction,
                'confidence': ensemble_confidence,
                'recommendation': recommendation,
                'insights': insights,
                'model_predictions': predictions
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'prediction': 0.5,
                'confidence': 50,
                'recommendation': 'ERROR',
                'insights': [f'Prediction failed: {str(e)}']
            }
    
    def _generate_prediction_insights(self, features, predictions):
        """Generate insights from predictions"""
        insights = []
        
        # Check feature patterns
        if features.get('rsi', 50) > 70:
            insights.append("⚠️ RSI indicates overbought conditions")
        elif features.get('rsi', 50) < 30:
            insights.append("✅ RSI indicates oversold opportunity")
        
        if features.get('iv_skew', 0) > 5:
            insights.append("📊 High IV skew suggests volatility opportunity")
        
        if features.get('last_3_trades_winrate', 0) > 0.7:
            insights.append("🔥 Recent winning streak detected")
        elif features.get('last_3_trades_winrate', 0) < 0.3:
            insights.append("⚠️ Recent losing streak - trade cautiously")
        
        # Model agreement
        model_agreement = np.std(list(predictions.values()))
        if model_agreement < 0.1:
            insights.append("✅ All models strongly agree on prediction")
        elif model_agreement > 0.3:
            insights.append("⚠️ Models show disagreement - higher risk")
        
        return insights
    
    def generate_learning_insights(self):
        """Generate insights from learned patterns"""
        print("\n📊 LEARNING INSIGHTS:")
        print("="*50)
        
        if self.winning_patterns:
            # Analyze winning patterns
            winning_df = pd.DataFrame(self.winning_patterns)
            print("\n✅ Winning Pattern Analysis:")
            print(f"   Total winning trades: {len(self.winning_patterns)}")
            print(f"   Average RSI: {winning_df['rsi'].mean():.2f}")
            print(f"   Average OI Ratio: {winning_df['oi_ratio'].mean():.2f}")
            print(f"   Best time of day: {self._get_best_time(winning_df)}")
            
        if self.losing_patterns:
            # Analyze losing patterns
            losing_df = pd.DataFrame(self.losing_patterns)
            print("\n❌ Losing Pattern Analysis:")
            print(f"   Total losing trades: {len(self.losing_patterns)}")
            print(f"   Common RSI range: {losing_df['rsi'].mean():.2f}")
            print(f"   Risk factors identified: {self._identify_risk_factors(losing_df)}")
        
        # Overall performance
        if self.performance_history:
            total_trades = len(self.performance_history)
            profitable_trades = sum(1 for t in self.performance_history if t['profit'] > 0)
            win_rate = profitable_trades / total_trades * 100
            
            print(f"\n📈 Overall Performance:")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Total Trades Analyzed: {total_trades}")
            
            # Feature importance
            if self.models['trend_predictor']:
                print(f"\n🎯 Most Important Features:")
                feature_imp = self.models['trend_predictor']['feature_importance']
                for _, row in feature_imp.head(5).iterrows():
                    print(f"   • {row['feature']}: {row['importance']:.3f}")
        
        print("="*50)
    
    def _get_best_time(self, df):
        """Identify best trading time"""
        df['hour'] = df['time_of_day'].apply(lambda x: int(x * 6.25 + 9))
        best_hour = df.groupby('hour').size().idxmax()
        return f"{best_hour}:00 - {best_hour+1}:00"
    
    def _identify_risk_factors(self, df):
        """Identify common risk factors in losing trades"""
        risk_factors = []
        
        if df['rsi'].mean() > 65:
            risk_factors.append("High RSI (overbought)")
        if df['iv_skew'].mean() < -5:
            risk_factors.append("Negative IV skew")
        if df['days_to_expiry'].mean() < 3:
            risk_factors.append("Too close to expiry")
        
        return ", ".join(risk_factors) if risk_factors else "Multiple factors"
    
    def save_models(self):
        """Save trained models to disk"""
        for model_name, model_data in self.models.items():
            if model_data:
                model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
                scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
                
                joblib.dump(model_data['model'], model_path)
                joblib.dump(model_data['scaler'], scaler_path)
                
                print(f"💾 Saved {model_name} model")
    
    def load_models(self):
        """Load existing models from disk"""
        for model_name in self.models.keys():
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[model_name] = {
                    'model': joblib.load(model_path),
                    'scaler': joblib.load(scaler_path),
                    'accuracy': 0.0,  # Will be updated on next training
                    'feature_importance': pd.DataFrame()
                }
                print(f"📂 Loaded {model_name} model")
    
    def save_learning_state(self):
        """Save current learning state"""
        state = {
            'performance_history': self.performance_history[-1000:],  # Keep last 1000
            'winning_patterns': self.winning_patterns[-500:],  # Keep last 500
            'losing_patterns': self.losing_patterns[-500:],
            'last_updated': datetime.now().isoformat()
        }
        
        state_path = os.path.join(self.model_dir, "learning_state.json")
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def get_strategy_recommendations(self):
        """Get strategy recommendations based on learning"""
        recommendations = []
        
        if len(self.winning_patterns) > 10:
            winning_df = pd.DataFrame(self.winning_patterns)
            
            # Time-based recommendations
            best_hour = self._get_best_time(winning_df)
            recommendations.append(f"🕐 Best trading time: {best_hour}")
            
            # Market condition recommendations
            avg_rsi = winning_df['rsi'].mean()
            if avg_rsi < 40:
                recommendations.append("📈 Focus on oversold conditions (RSI < 40)")
            elif avg_rsi > 60:
                recommendations.append("📉 Look for overbought reversals (RSI > 60)")
            
            # Moneyness recommendations
            avg_moneyness = winning_df['moneyness'].mean()
            if avg_moneyness < 0.98:
                recommendations.append("🎯 ITM options showing better success")
            elif avg_moneyness > 1.02:
                recommendations.append("🎯 OTM options with higher returns")
            
            # Risk recommendations
            if self.performance_history:
                recent_performance = self._get_recent_performance(10)
                if recent_performance < 0.4:
                    recommendations.append("⚠️ Reduce position size - low win rate")
                elif recent_performance > 0.7:
                    recommendations.append("✅ Consider increasing position size")
        
        return recommendations
    
    def should_take_trade(self, trade_setup):
        """Decision function using ML models"""
        # Extract features
        features = self.extract_trade_features(trade_setup)
        
        # Get prediction
        prediction = self.predict_trade_outcome(features)
        
        # Decision logic
        should_trade = (
            prediction['prediction'] > 0.6 and
            prediction['confidence'] > 70
        )
        
        return {
            'decision': should_trade,
            'ml_confidence': prediction['confidence'],
            'recommendation': prediction['recommendation'],
            'insights': prediction['insights'],
            'adjusted_position_size': self._calculate_position_size(prediction)
        }
    
    def _calculate_position_size(self, prediction):
        """Calculate position size based on ML confidence"""
        base_size = 1.0
        
        # Adjust based on confidence
        if prediction['confidence'] > 85:
            size_multiplier = 1.5
        elif prediction['confidence'] > 75:
            size_multiplier = 1.2
        elif prediction['confidence'] < 60:
            size_multiplier = 0.5
        else:
            size_multiplier = 1.0
        
        # Adjust based on recent performance
        recent_winrate = self._get_recent_performance(5)
        if recent_winrate < 0.3:
            size_multiplier *= 0.7  # Reduce size on losing streak
        elif recent_winrate > 0.8:
            size_multiplier *= 1.2  # Increase on winning streak
        
        return round(base_size * size_multiplier, 2)

# Integration function for your existing code
def integrate_ml_trader(trade_setup):
    """Function to integrate ML trader with existing system"""
    ml_trader = SelfLearningTrader()
    
    # Get ML decision
    ml_decision = ml_trader.should_take_trade(trade_setup)
    
    # Enhance existing trade setup with ML insights
    enhanced_setup = trade_setup.copy()
    enhanced_setup['ml_confidence'] = ml_decision['ml_confidence']
    enhanced_setup['ml_recommendation'] = ml_decision['recommendation']
    enhanced_setup['ml_insights'] = ml_decision['insights']
    enhanced_setup['suggested_lots'] = ml_decision['adjusted_position_size']
    
    return enhanced_setup, ml_decision['decision']

if __name__ == "__main__":
    # Test the self-learning module
    print("🧪 Testing Self-Learning AI Module...")
    
    ml_trader = SelfLearningTrader()
    
    # Simulate some historical trades for learning
    sample_trades = [
        {
            'rsi': 35, 'macd_signal': 1, 'oi_ratio': 1.3, 'iv_skew': -2,
            'moneyness': 0.98, 'days_to_expiry': 5, 'vix_level': 18,
            'pcr': 1.2, 'delta': 0.45, 'ai_confidence': 85,
            'timestamp': datetime.now() - timedelta(hours=2),
            'pnl': 1500  # Profit
        },
        {
            'rsi': 72, 'macd_signal': -1, 'oi_ratio': 0.8, 'iv_skew': 3,
            'moneyness': 1.05, 'days_to_expiry': 2, 'vix_level': 22,
            'pcr': 0.7, 'delta': 0.25, 'ai_confidence': 65,
            'timestamp': datetime.now() - timedelta(hours=4),
            'pnl': -800  # Loss
        }
    ]
    
    # Learn from trades
    for trade in sample_trades:
        ml_trader.learn_from_trade(trade)
    
    # Get recommendations
    recommendations = ml_trader.get_strategy_recommendations()
    print("\n🎯 Strategy Recommendations:")
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n✅ Self-Learning AI Module ready for integration!")