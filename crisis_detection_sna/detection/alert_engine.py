"""
CrisisDetect+ Alert Engine
Real-time crisis alerting and notification system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import deque, defaultdict

from utils.config import CRISIS_THRESHOLDS
from utils.logger import get_logger

logger = get_logger("AlertEngine")

class AlertLevel(Enum):
    """Alert severity levels"""
    SAFE = "SAFE"
    MONITORED = "MONITORED"
    WARNING = "WARNING"
    DETECTED = "DETECTED"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """Types of alerts"""
    CRISIS_DETECTED = "CRISIS_DETECTED"
    SEVERITY_INCREASE = "SEVERITY_INCREASE"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    NETWORK_ANOMALY = "NETWORK_ANOMALY"
    SENTIMENT_SHIFT = "SENTIMENT_SHIFT"
    EMOTION_SURGE = "EMOTION_SURGE"

class CrisisAlert:
    """Individual crisis alert"""
    
    def __init__(self, alert_type: AlertType, crisis_type: str, severity_level: AlertLevel,
                 severity_score: float, message: str, metadata: Dict = None):
        self.alert_id = self._generate_alert_id()
        self.alert_type = alert_type
        self.crisis_type = crisis_type
        self.severity_level = severity_level
        self.severity_score = severity_score
        self.message = message
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.acknowledged = False
        self.resolved = False
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ALERT_{timestamp}_{np.random.randint(1000, 9999)}"
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'crisis_type': self.crisis_type,
            'severity_level': self.severity_level.value,
            'severity_score': self.severity_score,
            'message': self.message,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved
        }
    
    def acknowledge(self):
        """Acknowledge the alert"""
        self.acknowledged = True
        self.metadata['acknowledged_at'] = datetime.now().isoformat()
    
    def resolve(self):
        """Resolve the alert"""
        self.resolved = True
        self.metadata['resolved_at'] = datetime.now().isoformat()

class CrisisAlertEngine:
    """Crisis alert engine for real-time monitoring"""
    
    def __init__(self, alert_history_size: int = 1000):
        self.thresholds = CRISIS_THRESHOLDS
        self.alert_history = deque(maxlen=alert_history_size)
        self.active_alerts = {}
        self.alert_callbacks = []
        self.crisis_history = defaultdict(list)
        self.last_severity_scores = {}
        
        # Alert suppression settings
        self.suppression_window = timedelta(minutes=30)
        self.last_alerts = defaultdict(datetime)
        
    def register_alert_callback(self, callback: Callable[[CrisisAlert], None]):
        """Register callback function for alert notifications"""
        self.alert_callbacks.append(callback)
        logger.info(f"📞 Registered alert callback: {callback.__name__}")
    
    def process_crisis_results(self, crisis_results: Dict[str, Dict]) -> List[CrisisAlert]:
        """Process crisis detection results and generate alerts"""
        logger.info("🚨 Processing crisis results for alerts")
        
        new_alerts = []
        
        for crisis_type, result in crisis_results.items():
            if 'error' in result:
                continue
            
            # Check for crisis detection alerts
            crisis_alerts = self._check_crisis_detection(crisis_type, result)
            new_alerts.extend(crisis_alerts)
            
            # Check for severity change alerts
            severity_alerts = self._check_severity_changes(crisis_type, result)
            new_alerts.extend(severity_alerts)
            
            # Check for anomaly alerts
            anomaly_alerts = self._check_anomalies(crisis_type, result)
            new_alerts.extend(anomaly_alerts)
            
            # Update history
            self._update_crisis_history(crisis_type, result)
        
        # Process and store alerts
        for alert in new_alerts:
            self._process_alert(alert)
        
        logger.info(f"🔔 Generated {len(new_alerts)} new alerts")
        
        return new_alerts
    
    def _check_crisis_detection(self, crisis_type: str, result: Dict) -> List[CrisisAlert]:
        """Check for crisis detection alerts"""
        alerts = []
        
        severity_score = result.get('severity_score', 0.0)
        severity_level = result.get('severity_level', 'SAFE')
        confidence = result.get('confidence', 0.0)
        
        # Crisis detected alert
        if severity_level in ['DETECTED', 'WARNING'] and confidence > 0.5:
            
            # Check if we should suppress this alert
            if self._should_suppress_alert(crisis_type, AlertType.CRISIS_DETECTED):
                return alerts
            
            alert_level = AlertLevel.DETECTED if severity_level == 'DETECTED' else AlertLevel.WARNING
            
            message = f"{crisis_type.title()} crisis {severity_level.lower()} with severity {severity_score:.3f}"
            
            metadata = {
                'factors': result.get('factors', {}),
                'confidence': confidence,
                'data_points': result.get('data_points', 0),
                'network_nodes': result.get('network_nodes', 0)
            }
            
            alert = CrisisAlert(
                alert_type=AlertType.CRISIS_DETECTED,
                crisis_type=crisis_type,
                severity_level=alert_level,
                severity_score=severity_score,
                message=message,
                metadata=metadata
            )
            
            alerts.append(alert)
            
            # Update suppression tracking
            self.last_alerts[(crisis_type, AlertType.CRISIS_DETECTED)] = datetime.now()
        
        return alerts
    
    def _check_severity_changes(self, crisis_type: str, result: Dict) -> List[CrisisAlert]:
        """Check for severity increase alerts"""
        alerts = []
        
        current_score = result.get('severity_score', 0.0)
        previous_score = self.last_severity_scores.get(crisis_type, 0.0)
        
        # Check for significant severity increase
        if previous_score > 0 and current_score > previous_score:
            increase = current_score - previous_score
            
            # Alert if increase is significant (>0.1) and current score is above monitoring threshold
            if increase > 0.1 and current_score > self.thresholds['monitored']:
                
                if self._should_suppress_alert(crisis_type, AlertType.SEVERITY_INCREASE):
                    return alerts
                
                message = f"{crisis_type.title()} crisis severity increased by {increase:.3f} (from {previous_score:.3f} to {current_score:.3f})"
                
                alert_level = AlertLevel.WARNING if current_score > self.thresholds['warning'] else AlertLevel.MONITORED
                
                metadata = {
                    'previous_score': previous_score,
                    'current_score': current_score,
                    'increase': increase,
                    'trend': 'increasing'
                }
                
                alert = CrisisAlert(
                    alert_type=AlertType.SEVERITY_INCREASE,
                    crisis_type=crisis_type,
                    severity_level=alert_level,
                    severity_score=current_score,
                    message=message,
                    metadata=metadata
                )
                
                alerts.append(alert)
                self.last_alerts[(crisis_type, AlertType.SEVERITY_INCREASE)] = datetime.now()
        
        # Update last score
        self.last_severity_scores[crisis_type] = current_score
        
        return alerts
    
    def _check_anomalies(self, crisis_type: str, result: Dict) -> List[CrisisAlert]:
        """Check for various anomaly alerts"""
        alerts = []
        
        factors = result.get('factors', {})
        
        # Volume spike alert
        volume_factor = factors.get('volume_factor', 0.0)
        if volume_factor > 0.8:  # High volume threshold
            
            if not self._should_suppress_alert(crisis_type, AlertType.VOLUME_SPIKE):
                message = f"High volume detected for {crisis_type} crisis (factor: {volume_factor:.3f})"
                
                alert = CrisisAlert(
                    alert_type=AlertType.VOLUME_SPIKE,
                    crisis_type=crisis_type,
                    severity_level=AlertLevel.MONITORED,
                    severity_score=volume_factor,
                    message=message,
                    metadata={'volume_factor': volume_factor}
                )
                
                alerts.append(alert)
                self.last_alerts[(crisis_type, AlertType.VOLUME_SPIKE)] = datetime.now()
        
        # Sentiment shift alert
        sentiment_factor = factors.get('sentiment_factor', 0.0)
        if sentiment_factor > 0.7:  # High negative sentiment
            
            if not self._should_suppress_alert(crisis_type, AlertType.SENTIMENT_SHIFT):
                message = f"Significant negative sentiment shift in {crisis_type} (factor: {sentiment_factor:.3f})"
                
                alert = CrisisAlert(
                    alert_type=AlertType.SENTIMENT_SHIFT,
                    crisis_type=crisis_type,
                    severity_level=AlertLevel.MONITORED,
                    severity_score=sentiment_factor,
                    message=message,
                    metadata={'sentiment_factor': sentiment_factor}
                )
                
                alerts.append(alert)
                self.last_alerts[(crisis_type, AlertType.SENTIMENT_SHIFT)] = datetime.now()
        
        # Emotion surge alert
        emotion_factor = factors.get('emotion_factor', 0.0)
        if emotion_factor > 0.8:  # High crisis emotions
            
            if not self._should_suppress_alert(crisis_type, AlertType.EMOTION_SURGE):
                message = f"High crisis emotion intensity in {crisis_type} (factor: {emotion_factor:.3f})"
                
                alert = CrisisAlert(
                    alert_type=AlertType.EMOTION_SURGE,
                    crisis_type=crisis_type,
                    severity_level=AlertLevel.MONITORED,
                    severity_score=emotion_factor,
                    message=message,
                    metadata={'emotion_factor': emotion_factor}
                )
                
                alerts.append(alert)
                self.last_alerts[(crisis_type, AlertType.EMOTION_SURGE)] = datetime.now()
        
        return alerts
    
    def _should_suppress_alert(self, crisis_type: str, alert_type: AlertType) -> bool:
        """Check if alert should be suppressed due to recent similar alert"""
        
        last_alert_time = self.last_alerts.get((crisis_type, alert_type))
        
        if last_alert_time:
            time_since_last = datetime.now() - last_alert_time
            return time_since_last < self.suppression_window
        
        return False
    
    def _process_alert(self, alert: CrisisAlert):
        """Process and store alert"""
        
        # Add to history
        self.alert_history.append(alert)
        
        # Add to active alerts if not resolved
        if not alert.resolved:
            self.active_alerts[alert.alert_id] = alert
        
        # Log alert
        logger.log_crisis_event(
            alert.crisis_type,
            f"{alert.alert_type.value}: {alert.message}",
            alert.severity_score
        )
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"❌ Alert callback failed: {e}")
    
    def _update_crisis_history(self, crisis_type: str, result: Dict):
        """Update crisis history for trend analysis"""
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'severity_score': result.get('severity_score', 0.0),
            'severity_level': result.get('severity_level', 'SAFE'),
            'confidence': result.get('confidence', 0.0),
            'factors': result.get('factors', {})
        }
        
        self.crisis_history[crisis_type].append(history_entry)
        
        # Keep only recent history (last 100 entries)
        if len(self.crisis_history[crisis_type]) > 100:
            self.crisis_history[crisis_type] = self.crisis_history[crisis_type][-100:]
    
    def get_active_alerts(self, crisis_type: Optional[str] = None) -> List[Dict]:
        """Get currently active alerts"""
        
        active = list(self.active_alerts.values())
        
        if crisis_type:
            active = [alert for alert in active if alert.crisis_type == crisis_type]
        
        return [alert.to_dict() for alert in active]
    
    def get_alert_history(self, crisis_type: Optional[str] = None, 
                         hours: int = 24) -> List[Dict]:
        """Get alert history for specified time period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [alert for alert in self.alert_history 
                        if alert.timestamp >= cutoff_time]
        
        if crisis_type:
            recent_alerts = [alert for alert in recent_alerts 
                           if alert.crisis_type == crisis_type]
        
        return [alert.to_dict() for alert in recent_alerts]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledge()
            logger.info(f"✅ Alert {alert_id} acknowledged")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolve()
            del self.active_alerts[alert_id]
            logger.info(f"✅ Alert {alert_id} resolved")
            return True
        
        return False
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alert system status"""
        
        active_count = len(self.active_alerts)
        recent_count = len(self.get_alert_history(hours=24))
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            severity_counts[alert.severity_level.value] += 1
        
        # Count by type
        type_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            type_counts[alert.alert_type.value] += 1
        
        summary = {
            'active_alerts': active_count,
            'recent_alerts_24h': recent_count,
            'severity_breakdown': dict(severity_counts),
            'type_breakdown': dict(type_counts),
            'last_update': datetime.now().isoformat(),
            'suppression_window_minutes': self.suppression_window.total_seconds() / 60
        }
        
        return summary

# Default alert callback functions
def console_alert_callback(alert: CrisisAlert):
    """Simple console alert callback"""
    print(f"🚨 ALERT: {alert.message}")

def log_alert_callback(alert: CrisisAlert):
    """Log alert callback"""
    logger.info(f"🚨 {alert.alert_type.value}: {alert.message}")

if __name__ == "__main__":
    # Test alert engine
    engine = CrisisAlertEngine()
    
    # Register callbacks
    engine.register_alert_callback(console_alert_callback)
    engine.register_alert_callback(log_alert_callback)
    
    # Test with sample crisis results
    sample_results = {
        'health': {
            'severity_score': 0.45,
            'severity_level': 'DETECTED',
            'confidence': 0.8,
            'factors': {
                'sentiment_factor': 0.6,
                'emotion_factor': 0.7,
                'volume_factor': 0.3,
                'network_factor': 0.4
            }
        }
    }
    
    alerts = engine.process_crisis_results(sample_results)
    summary = engine.get_alert_summary()
    
    print(f"Generated {len(alerts)} alerts")
    print(f"Alert summary: {summary}")
    
    logger.info("✅ Alert engine tested successfully!")
