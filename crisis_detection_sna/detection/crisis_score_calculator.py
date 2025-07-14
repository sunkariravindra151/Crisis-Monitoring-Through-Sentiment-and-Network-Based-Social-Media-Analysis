"""
CrisisDetect+ Crisis Score Calculator
Multi-factor crisis severity scoring system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import networkx as nx

from utils.config import CRISIS_TYPES, CRISIS_THRESHOLDS
from utils.logger import get_logger

logger = get_logger("CrisisScoreCalculator")

class CrisisScoreCalculator:
    """Calculate crisis severity scores using multi-factor analysis"""
    
    def __init__(self):
        self.crisis_types = CRISIS_TYPES
        self.thresholds = CRISIS_THRESHOLDS
        
    def calculate_crisis_severity(self, df: pd.DataFrame, network: nx.Graph, 
                                crisis_type: str, network_metrics: Dict = None) -> Dict[str, Union[float, str]]:
        """Calculate comprehensive crisis severity score"""
        logger.info(f"🚨 Calculating crisis severity for {crisis_type}")
        
        if df.empty:
            logger.warning(f"⚠️ No data available for {crisis_type}")
            return self._empty_crisis_result(crisis_type)
        
        # Get crisis-specific weights
        weights = self.crisis_types[crisis_type]['severity_weights']
        
        # Calculate individual factors
        sentiment_factor = self._calculate_sentiment_factor(df)
        emotion_factor = self._calculate_emotion_factor(df)
        volume_factor = self._calculate_volume_factor(df, crisis_type)
        network_factor = self._calculate_network_factor(network, network_metrics)
        
        # Calculate weighted severity score
        severity_score = (
            weights['sentiment'] * sentiment_factor +
            weights['emotion'] * emotion_factor +
            weights['volume'] * volume_factor +
            weights['network'] * network_factor
        )
        
        # Determine severity level
        severity_level = self._determine_severity_level(severity_score)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(df, network, severity_score)
        
        # Prepare result
        result = {
            'crisis_type': crisis_type,
            'severity_score': float(severity_score),
            'severity_level': severity_level,
            'confidence': float(confidence),
            'factors': {
                'sentiment_factor': float(sentiment_factor),
                'emotion_factor': float(emotion_factor),
                'volume_factor': float(volume_factor),
                'network_factor': float(network_factor)
            },
            'weights': weights,
            'threshold_used': self._get_threshold_for_level(severity_level),
            'calculation_timestamp': datetime.now().isoformat(),
            'data_points': len(df),
            'network_nodes': network.number_of_nodes() if network else 0
        }
        
        logger.log_crisis_event(crisis_type, f"Severity calculated: {severity_level}", severity_score)
        
        return result
    
    def _calculate_sentiment_factor(self, df: pd.DataFrame) -> float:
        """Calculate sentiment-based crisis factor"""
        
        if 'sentiment_compound' not in df.columns:
            logger.warning("⚠️ No sentiment data available")
            return 0.0
        
        sentiments = df['sentiment_compound'].dropna()
        
        if sentiments.empty:
            return 0.0
        
        # Crisis indicators from sentiment
        avg_sentiment = sentiments.mean()
        sentiment_volatility = sentiments.std()
        negative_ratio = (sentiments < -0.1).mean()
        extreme_negative_ratio = (sentiments < -0.5).mean()
        
        # Combine sentiment indicators
        sentiment_intensity = abs(avg_sentiment)
        volatility_factor = min(1.0, sentiment_volatility / 0.5)  # Normalize volatility
        negative_factor = negative_ratio
        extreme_factor = extreme_negative_ratio * 2.0  # Double weight for extreme negativity
        
        # Weighted combination
        sentiment_factor = (
            0.3 * sentiment_intensity +
            0.2 * volatility_factor +
            0.3 * negative_factor +
            0.2 * extreme_factor
        )
        
        return min(1.0, sentiment_factor)
    
    def _calculate_emotion_factor(self, df: pd.DataFrame) -> float:
        """Calculate emotion-based crisis factor"""
        
        if 'emotion_crisis_intensity' not in df.columns:
            logger.warning("⚠️ No emotion data available")
            return 0.0
        
        emotions = df['emotion_crisis_intensity'].dropna()
        
        if emotions.empty:
            return 0.0
        
        # Crisis emotion indicators
        avg_crisis_intensity = emotions.mean()
        emotion_volatility = emotions.std()
        high_intensity_ratio = (emotions > 0.7).mean()
        
        # Check for specific crisis emotions
        crisis_emotion_factor = 0.0
        crisis_emotions = ['fear', 'anger', 'sadness', 'disgust']
        
        for emotion in crisis_emotions:
            emotion_col = f'emotion_{emotion}'
            if emotion_col in df.columns:
                emotion_presence = df[emotion_col].mean()
                crisis_emotion_factor += emotion_presence
        
        crisis_emotion_factor = min(1.0, crisis_emotion_factor)
        
        # Combine emotion indicators
        emotion_factor = (
            0.4 * avg_crisis_intensity +
            0.2 * min(1.0, emotion_volatility) +
            0.2 * high_intensity_ratio +
            0.2 * crisis_emotion_factor
        )
        
        return min(1.0, emotion_factor)
    
    def _calculate_volume_factor(self, df: pd.DataFrame, crisis_type: str) -> float:
        """Calculate volume-based crisis factor"""
        
        current_volume = len(df)
        
        # Get baseline volume for crisis type (could be historical average)
        baseline_volume = self._get_baseline_volume(crisis_type)
        
        if baseline_volume == 0:
            # If no baseline, use relative volume within current data
            total_posts = current_volume
            volume_factor = min(1.0, current_volume / max(1000, total_posts))
        else:
            # Calculate volume spike
            volume_ratio = current_volume / baseline_volume
            volume_factor = min(1.0, (volume_ratio - 1.0) / 4.0)  # Normalize spike
        
        # Consider temporal concentration
        if 'datetime' in df.columns:
            temporal_factor = self._calculate_temporal_concentration(df)
            volume_factor = 0.7 * volume_factor + 0.3 * temporal_factor
        
        return max(0.0, volume_factor)
    
    def _calculate_network_factor(self, network: nx.Graph, network_metrics: Dict = None) -> float:
        """Calculate network-based crisis factor"""
        
        if not network or network.number_of_nodes() == 0:
            logger.warning("⚠️ No network data available")
            return 0.0
        
        network_factor = 0.0
        
        # Network density (higher density can indicate crisis clustering)
        density = nx.density(network)
        density_factor = min(1.0, density * 2.0)  # Normalize density
        
        # Network fragmentation (more components = more fragmentation)
        if nx.is_connected(network):
            fragmentation_factor = 0.0
        else:
            num_components = nx.number_connected_components(network)
            fragmentation_factor = min(1.0, (num_components - 1) / network.number_of_nodes())
        
        # Clustering coefficient (higher clustering in crisis)
        clustering = nx.average_clustering(network, weight='weight')
        clustering_factor = clustering
        
        # Use network metrics if available
        if network_metrics:
            modularity = network_metrics.get('modularity', 0.0)
            modularity_factor = 1.0 - modularity  # Lower modularity = higher crisis indicator
            
            # Centralization (higher centralization = crisis leaders)
            centralization = network_metrics.get('degree_centralization', 0.0)
            centralization_factor = centralization
            
            # Combine all network factors
            network_factor = (
                0.25 * density_factor +
                0.20 * fragmentation_factor +
                0.20 * clustering_factor +
                0.20 * modularity_factor +
                0.15 * centralization_factor
            )
        else:
            # Basic network factors only
            network_factor = (
                0.4 * density_factor +
                0.3 * fragmentation_factor +
                0.3 * clustering_factor
            )
        
        return min(1.0, network_factor)
    
    def _calculate_temporal_concentration(self, df: pd.DataFrame) -> float:
        """Calculate temporal concentration of posts"""
        
        if 'datetime' not in df.columns:
            return 0.0
        
        try:
            timestamps = pd.to_datetime(df['datetime']).dropna()
            
            if len(timestamps) < 2:
                return 0.0
            
            # Calculate time spans
            time_span = (timestamps.max() - timestamps.min()).total_seconds() / 3600  # Hours
            
            if time_span == 0:
                return 1.0  # All posts at same time = maximum concentration
            
            # Calculate posts per hour
            posts_per_hour = len(timestamps) / time_span
            
            # Normalize (assuming 10 posts/hour is high concentration)
            concentration = min(1.0, posts_per_hour / 10.0)
            
            return concentration
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating temporal concentration: {e}")
            return 0.0
    
    def _get_baseline_volume(self, crisis_type: str) -> int:
        """Get baseline volume for crisis type (placeholder for historical data)"""
        
        # Placeholder baseline volumes (in real system, this would come from historical data)
        baselines = {
            'health': 500,
            'technological': 300,
            'political': 800,
            'social': 600
        }
        
        return baselines.get(crisis_type, 400)
    
    def _determine_severity_level(self, severity_score: float) -> str:
        """Determine severity level based on score and thresholds"""
        
        if severity_score >= self.thresholds['detected']:
            return 'DETECTED'
        elif severity_score >= self.thresholds['warning']:
            return 'WARNING'
        elif severity_score >= self.thresholds['monitored']:
            return 'MONITORED'
        else:
            return 'SAFE'
    
    def _get_threshold_for_level(self, level: str) -> float:
        """Get threshold value for severity level"""
        
        threshold_mapping = {
            'DETECTED': self.thresholds['detected'],
            'WARNING': self.thresholds['warning'],
            'MONITORED': self.thresholds['monitored'],
            'SAFE': self.thresholds['safe']
        }
        
        return threshold_mapping.get(level, 0.0)
    
    def _calculate_confidence(self, df: pd.DataFrame, network: nx.Graph, severity_score: float) -> float:
        """Calculate confidence in the crisis detection"""
        
        confidence_factors = []
        
        # Data volume confidence
        data_volume = len(df)
        volume_confidence = min(1.0, data_volume / 100.0)  # Full confidence at 100+ posts
        confidence_factors.append(volume_confidence)
        
        # Network size confidence
        if network and network.number_of_nodes() > 0:
            network_confidence = min(1.0, network.number_of_nodes() / 50.0)  # Full confidence at 50+ nodes
            confidence_factors.append(network_confidence)
        
        # Data completeness confidence
        required_cols = ['sentiment_compound', 'emotion_crisis_intensity']
        available_cols = [col for col in required_cols if col in df.columns and not df[col].isna().all()]
        completeness_confidence = len(available_cols) / len(required_cols)
        confidence_factors.append(completeness_confidence)
        
        # Severity consistency confidence (higher scores = higher confidence)
        severity_confidence = min(1.0, severity_score * 2.0)
        confidence_factors.append(severity_confidence)
        
        # Overall confidence
        overall_confidence = np.mean(confidence_factors)
        
        return overall_confidence
    
    def _empty_crisis_result(self, crisis_type: str) -> Dict:
        """Return empty result for crisis with no data"""
        
        return {
            'crisis_type': crisis_type,
            'severity_score': 0.0,
            'severity_level': 'SAFE',
            'confidence': 0.0,
            'factors': {
                'sentiment_factor': 0.0,
                'emotion_factor': 0.0,
                'volume_factor': 0.0,
                'network_factor': 0.0
            },
            'weights': self.crisis_types[crisis_type]['severity_weights'],
            'threshold_used': self.thresholds['safe'],
            'calculation_timestamp': datetime.now().isoformat(),
            'data_points': 0,
            'network_nodes': 0,
            'error': 'No data available'
        }
    
    def compare_crisis_severities(self, crisis_results: Dict[str, Dict]) -> Dict[str, Union[List, str, float]]:
        """Compare and rank crisis severities"""
        
        valid_results = {crisis: result for crisis, result in crisis_results.items() 
                        if 'error' not in result}
        
        if not valid_results:
            return {'error': 'No valid crisis results to compare'}
        
        # Sort by severity score
        sorted_crises = sorted(valid_results.items(), 
                             key=lambda x: x[1]['severity_score'], 
                             reverse=True)
        
        # Create ranking
        ranking = []
        for i, (crisis_type, result) in enumerate(sorted_crises, 1):
            ranking.append({
                'rank': i,
                'crisis_type': crisis_type,
                'severity_score': result['severity_score'],
                'severity_level': result['severity_level'],
                'confidence': result['confidence']
            })
        
        # Overall statistics
        scores = [result['severity_score'] for result in valid_results.values()]
        
        comparison = {
            'ranking': ranking,
            'most_severe': sorted_crises[0][0] if sorted_crises else None,
            'highest_score': max(scores) if scores else 0.0,
            'average_score': np.mean(scores) if scores else 0.0,
            'total_detected': sum(1 for result in valid_results.values() 
                                if result['severity_level'] == 'DETECTED'),
            'total_warning': sum(1 for result in valid_results.values() 
                               if result['severity_level'] == 'WARNING'),
            'comparison_timestamp': datetime.now().isoformat()
        }
        
        return comparison

if __name__ == "__main__":
    # Test crisis score calculator
    calculator = CrisisScoreCalculator()
        
    # Create sample data
    sample_data = pd.DataFrame({
        'sentiment_compound': np.random.uniform(-1, 1, 100),
        'emotion_crisis_intensity': np.random.uniform(0, 1, 100),
        'datetime': pd.date_range('2024-01-01', periods=100, freq='H')
    })
    
    # Create sample network
    G = nx.erdos_renyi_graph(20, 0.3)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
    
    # Calculate crisis severity
    result = calculator.calculate_crisis_severity(sample_data, G, 'health')
    
    print(f"Crisis Type: {result['crisis_type']}")
    print(f"Severity Score: {result['severity_score']:.3f}")
    print(f"Severity Level: {result['severity_level']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    logger.info("✅ Crisis score calculator tested successfully!")

