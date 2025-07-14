"""
CrisisDetect+ Sentiment Classifier
BERT-based sentiment analysis for crisis detection
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

logger = get_logger("SentimentClassifier")

class CrisisSentimentClassifier:
    """BERT-based sentiment classifier optimized for crisis detection"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or NLP_CONFIG['sentiment_model']
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()
    
    def _setup_device(self, device: Optional[str]) -> str:
        """Setup computation device (GPU/CPU)"""
        if device:
            return device
        elif NLP_CONFIG['device'] == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info("🚀 Using GPU for sentiment analysis")
            else:
                device = 'cpu'
                logger.info("💻 Using CPU for sentiment analysis")
        else:
            device = NLP_CONFIG['device']
        
        return device
    
    def _load_model(self):
        """Load BERT model and tokenizer"""
        try:
            logger.log_model_loading(self.model_name, "loading")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline
            device_id = 0 if self.device == 'cuda' else -1
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_id,
                return_all_scores=True
            )
            
            logger.log_model_loading(self.model_name, "loaded")
            
        except Exception as e:
            logger.error(f"❌ Error loading sentiment model: {e}")
            raise
    
    def predict_single(self, text: str) -> Dict[str, Union[str, float]]:
        """Predict sentiment for a single text"""
        if not text or not isinstance(text, str):
            return {
                'label': 'NEUTRAL',
                'score': 0.5,
                'compound': 0.0,
                'confidence': 0.0
            }
        
        try:
            # Truncate text if too long
            if len(text) > NLP_CONFIG['max_length']:
                text = text[:NLP_CONFIG['max_length']]
            
            # Get prediction
            results = self.pipeline(text)
            
            # Process results (handle both single and multi-label outputs)
            if isinstance(results[0], list):
                # Multi-label output
                scores = {result['label']: result['score'] for result in results[0]}
            else:
                # Single label output
                scores = {results[0]['label']: results[0]['score']}
            
            # Determine primary sentiment
            primary_label = max(scores.keys(), key=lambda k: scores[k])
            primary_score = scores[primary_label]
            
            # Calculate compound score (-1 to 1)
            compound = self._calculate_compound_score(scores, primary_label)
            
            return {
                'label': primary_label,
                'score': primary_score,
                'compound': compound,
                'confidence': primary_score,
                'all_scores': scores
            }
            
        except Exception as e:
            logger.error(f"❌ Error predicting sentiment: {e}")
            return {
                'label': 'NEUTRAL',
                'score': 0.5,
                'compound': 0.0,
                'confidence': 0.0
            }
    
    def _calculate_compound_score(self, scores: Dict[str, float], primary_label: str) -> float:
        """Calculate compound sentiment score (-1 to 1)"""
        
        # Map labels to sentiment values
        sentiment_mapping = {
            'POSITIVE': 1.0,
            'NEGATIVE': -1.0,
            'NEUTRAL': 0.0,
            'POS': 1.0,
            'NEG': -1.0,
            'LABEL_0': -1.0,  # Often negative in some models
            'LABEL_1': 1.0,   # Often positive in some models
        }
        
        # Calculate weighted compound score
        compound = 0.0
        total_weight = 0.0
        
        for label, score in scores.items():
            sentiment_value = sentiment_mapping.get(label.upper(), 0.0)
            compound += sentiment_value * score
            total_weight += score
        
        # Normalize
        if total_weight > 0:
            compound = compound / total_weight
        
        # Ensure bounds
        compound = max(-1.0, min(1.0, compound))
        
        return compound
    
    def predict_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[Dict]:
        """Predict sentiment for a batch of texts"""
        batch_size = batch_size or NLP_CONFIG['batch_size']
        results = []
        
        logger.info(f"🔄 Processing {len(texts)} texts in batches of {batch_size}")
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                result = self.predict_single(text)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        logger.info(f"✅ Completed sentiment analysis for {len(texts)} texts")
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Analyze sentiment for entire DataFrame"""
        logger.info(f"📊 Starting sentiment analysis for {len(df)} records")
        
        # Extract texts
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # Predict sentiments
        results = self.predict_batch(texts)
        
        # Add results to DataFrame
        df_copy = df.copy()
        df_copy['sentiment_label'] = [r['label'] for r in results]
        df_copy['sentiment_score'] = [r['score'] for r in results]
        df_copy['sentiment_compound'] = [r['compound'] for r in results]
        df_copy['sentiment_confidence'] = [r['confidence'] for r in results]
        
        # Add sentiment statistics
        self._add_sentiment_stats(df_copy)
        
        logger.info("✅ Sentiment analysis completed")
        return df_copy
    
    def _add_sentiment_stats(self, df: pd.DataFrame):
        """Add sentiment statistics to DataFrame"""
        
        # Sentiment distribution
        sentiment_dist = df['sentiment_label'].value_counts(normalize=True)
        logger.info(f"📈 Sentiment Distribution: {sentiment_dist.to_dict()}")
        
        # Compound score statistics
        compound_stats = df['sentiment_compound'].describe()
        logger.info(f"📊 Compound Score Stats - Mean: {compound_stats['mean']:.3f}, Std: {compound_stats['std']:.3f}")
        
        # Crisis-relevant metrics
        negative_ratio = (df['sentiment_compound'] < -0.1).mean()
        positive_ratio = (df['sentiment_compound'] > 0.1).mean()
        neutral_ratio = (df['sentiment_compound'].abs() <= 0.1).mean()
        
        logger.info(f"🎯 Crisis Metrics - Negative: {negative_ratio:.3f}, Positive: {positive_ratio:.3f}, Neutral: {neutral_ratio:.3f}")
    
    def get_crisis_sentiment_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract crisis-specific sentiment indicators"""
        
        indicators = {
            'avg_sentiment': df['sentiment_compound'].mean(),
            'sentiment_volatility': df['sentiment_compound'].std(),
            'negative_ratio': (df['sentiment_compound'] < -0.1).mean(),
            'extreme_negative_ratio': (df['sentiment_compound'] < -0.5).mean(),
            'sentiment_skew': df['sentiment_compound'].skew(),
            'min_sentiment': df['sentiment_compound'].min(),
            'max_sentiment': df['sentiment_compound'].max()
        }
        
        return indicators

if __name__ == "__main__":
    # Test sentiment classifier
    classifier = CrisisSentimentClassifier()
    
    # Test single prediction
    test_texts = [
        "This is a terrible crisis situation!",
        "Everything is working perfectly fine.",
        "I'm not sure what to think about this."
    ]
    
    for text in test_texts:
        result = classifier.predict_single(text)
        print(f"Text: {text}")
        print(f"Result: {result}")
        print("-" * 50)
    
    logger.info("✅ Sentiment classifier tested successfully!")
