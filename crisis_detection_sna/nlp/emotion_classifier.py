"""
CrisisDetect+ Emotion Classifier
RoBERTa-based emotion analysis for crisis detection
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from utils.config import NLP_CONFIG
from utils.logger import get_logger

logger = get_logger("EmotionClassifier")

class CrisisEmotionClassifier:
    """RoBERTa-based emotion classifier for crisis analysis"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or NLP_CONFIG['emotion_model']
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Crisis-relevant emotion mapping
        self.crisis_emotions = {
            'fear': {'weight': 1.0, 'crisis_indicator': True},
            'anger': {'weight': 0.9, 'crisis_indicator': True},
            'sadness': {'weight': 0.8, 'crisis_indicator': True},
            'disgust': {'weight': 0.7, 'crisis_indicator': True},
            'surprise': {'weight': 0.3, 'crisis_indicator': False},
            'joy': {'weight': -0.5, 'crisis_indicator': False},
            'neutral': {'weight': 0.0, 'crisis_indicator': False}
        }
        
        self._load_model()
    
    def _setup_device(self, device: Optional[str]) -> str:
        """Setup computation device (GPU/CPU)"""
        if device:
            return device
        elif NLP_CONFIG['device'] == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info("🚀 Using GPU for emotion analysis")
            else:
                device = 'cpu'
                logger.info("💻 Using CPU for emotion analysis")
        else:
            device = NLP_CONFIG['device']
        
        return device
    
    def _load_model(self):
        """Load RoBERTa emotion model"""
        try:
            logger.log_model_loading(self.model_name, "loading")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline
            device_id = 0 if self.device == 'cuda' else -1
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_id,
                return_all_scores=True
            )
            
            logger.log_model_loading(self.model_name, "loaded")
            
        except Exception as e:
            logger.error(f"❌ Error loading emotion model: {e}")
            raise
    
    def predict_single(self, text: str) -> Dict[str, Union[str, float, Dict]]:
        """Predict emotions for a single text"""
        if not text or not isinstance(text, str):
            return {
                'primary_emotion': 'neutral',
                'primary_score': 0.5,
                'crisis_intensity': 0.0,
                'emotion_scores': {},
                'crisis_emotions_total': 0.0
            }
        
        try:
            # Truncate text if too long
            if len(text) > NLP_CONFIG['max_length']:
                text = text[:NLP_CONFIG['max_length']]
            
            # Get prediction
            results = self.pipeline(text)
            
            # Process results
            emotion_scores = {}
            if isinstance(results, list) and len(results) > 0:
                # Handle nested list structure from pipeline
                if isinstance(results[0], list):
                    results = results[0]

                for result in results:
                    if isinstance(result, dict) and 'label' in result and 'score' in result:
                        emotion = result['label'].lower()
                        score = result['score']
                        emotion_scores[emotion] = score
            
            # Find primary emotion
            primary_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
            primary_score = emotion_scores[primary_emotion]
            
            # Calculate crisis intensity
            crisis_intensity = self._calculate_crisis_intensity(emotion_scores)
            
            # Calculate total crisis emotions
            crisis_emotions_total = self._calculate_crisis_emotions_total(emotion_scores)
            
            return {
                'primary_emotion': primary_emotion,
                'primary_score': primary_score,
                'crisis_intensity': crisis_intensity,
                'emotion_scores': emotion_scores,
                'crisis_emotions_total': crisis_emotions_total
            }
            
        except Exception as e:
            logger.error(f"❌ Error predicting emotions: {e}")
            return {
                'primary_emotion': 'neutral',
                'primary_score': 0.5,
                'crisis_intensity': 0.0,
                'emotion_scores': {},
                'crisis_emotions_total': 0.0
            }
    
    def _calculate_crisis_intensity(self, emotion_scores: Dict[str, float]) -> float:
        """Calculate crisis intensity based on emotion weights"""
        intensity = 0.0
        total_weight = 0.0
        
        for emotion, score in emotion_scores.items():
            if emotion in self.crisis_emotions:
                weight = self.crisis_emotions[emotion]['weight']
                intensity += score * weight
                total_weight += abs(weight)
        
        # Normalize
        if total_weight > 0:
            intensity = intensity / total_weight
        
        # Ensure positive value for crisis intensity
        intensity = max(0.0, intensity)
        
        return intensity
    
    def _calculate_crisis_emotions_total(self, emotion_scores: Dict[str, float]) -> float:
        """Calculate total score for crisis-indicating emotions"""
        crisis_total = 0.0
        
        for emotion, score in emotion_scores.items():
            if emotion in self.crisis_emotions and self.crisis_emotions[emotion]['crisis_indicator']:
                crisis_total += score
        
        return crisis_total
    
    def predict_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[Dict]:
        """Predict emotions for a batch of texts"""
        batch_size = batch_size or NLP_CONFIG['batch_size']
        results = []
        
        logger.info(f"🔄 Processing {len(texts)} texts for emotion analysis")
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Emotion Analysis"):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                result = self.predict_single(text)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        logger.info(f"✅ Completed emotion analysis for {len(texts)} texts")
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Analyze emotions for entire DataFrame"""
        logger.info(f"📊 Starting emotion analysis for {len(df)} records")
        
        # Extract texts
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # Predict emotions
        results = self.predict_batch(texts)
        
        # Add results to DataFrame
        df_copy = df.copy()
        df_copy['emotion_primary'] = [r['primary_emotion'] for r in results]
        df_copy['emotion_score'] = [r['primary_score'] for r in results]
        df_copy['emotion_crisis_intensity'] = [r['crisis_intensity'] for r in results]
        df_copy['emotion_crisis_total'] = [r['crisis_emotions_total'] for r in results]
        
        # Add individual emotion scores
        all_emotions = set()
        for r in results:
            all_emotions.update(r['emotion_scores'].keys())
        
        for emotion in all_emotions:
            df_copy[f'emotion_{emotion}'] = [r['emotion_scores'].get(emotion, 0.0) for r in results]
        
        # Add emotion statistics
        self._add_emotion_stats(df_copy)
        
        logger.info("✅ Emotion analysis completed")
        return df_copy
    
    def _add_emotion_stats(self, df: pd.DataFrame):
        """Add emotion statistics to DataFrame"""
        
        # Primary emotion distribution
        emotion_dist = df['emotion_primary'].value_counts(normalize=True)
        logger.info(f"😊 Emotion Distribution: {emotion_dist.to_dict()}")
        
        # Crisis intensity statistics
        crisis_stats = df['emotion_crisis_intensity'].describe()
        logger.info(f"🚨 Crisis Intensity Stats - Mean: {crisis_stats['mean']:.3f}, Std: {crisis_stats['std']:.3f}")
        
        # Crisis emotion ratios
        crisis_emotion_ratio = df['emotion_crisis_total'].mean()
        logger.info(f"⚠️ Average Crisis Emotion Ratio: {crisis_emotion_ratio:.3f}")
    
    def get_crisis_emotion_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract crisis-specific emotion indicators"""
        
        indicators = {
            'avg_crisis_intensity': df['emotion_crisis_intensity'].mean(),
            'crisis_intensity_volatility': df['emotion_crisis_intensity'].std(),
            'crisis_emotions_ratio': df['emotion_crisis_total'].mean(),
            'fear_ratio': df.get('emotion_fear', pd.Series([0])).mean(),
            'anger_ratio': df.get('emotion_anger', pd.Series([0])).mean(),
            'sadness_ratio': df.get('emotion_sadness', pd.Series([0])).mean(),
            'joy_ratio': df.get('emotion_joy', pd.Series([0])).mean(),
            'max_crisis_intensity': df['emotion_crisis_intensity'].max()
        }
        
        return indicators
    
    def get_emotion_distribution(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Get detailed emotion distribution for visualization"""
        
        distribution = {}
        
        # Primary emotion counts
        emotion_counts = df['emotion_primary'].value_counts()
        total_count = len(df)
        
        for emotion, count in emotion_counts.items():
            distribution[emotion] = {
                'count': int(count),
                'percentage': float(count / total_count * 100),
                'is_crisis_emotion': self.crisis_emotions.get(emotion, {}).get('crisis_indicator', False)
            }
        
        return distribution

if __name__ == "__main__":
    # Test emotion classifier
    classifier = CrisisEmotionClassifier()
    
    # Test single prediction
    test_texts = [
        "I'm terrified about this crisis situation!",
        "This makes me so angry and frustrated!",
        "I'm feeling happy and optimistic today.",
        "This is a neutral statement about the weather."
    ]
    
    for text in test_texts:
        result = classifier.predict_single(text)
        print(f"Text: {text}")
        print(f"Primary Emotion: {result['primary_emotion']} ({result['primary_score']:.3f})")
        print(f"Crisis Intensity: {result['crisis_intensity']:.3f}")
        print(f"Emotion Scores: {result['emotion_scores']}")
        print("-" * 50)
    
    logger.info("✅ Emotion classifier tested successfully!")
