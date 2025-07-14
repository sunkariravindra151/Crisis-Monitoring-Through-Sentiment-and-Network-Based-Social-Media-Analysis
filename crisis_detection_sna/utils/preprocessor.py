"""
CrisisDetect+ Data Preprocessor
Handles data cleaning and preparation for crisis-specific analysis
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import unicodedata

from utils.logger import get_logger

logger = get_logger("Preprocessor")

class CrisisDataPreprocessor:
    """Preprocessor for crisis-specific datasets"""
    
    def __init__(self):
        self.text_cleaning_patterns = {
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'mentions': re.compile(r'@[A-Za-z0-9_]+'),
            'hashtags': re.compile(r'#[A-Za-z0-9_]+'),
            'extra_spaces': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s]'),
            'numbers': re.compile(r'\d+')
        }
    
    def load_crisis_data(self, crisis_type: str, data_path: Path) -> pd.DataFrame:
        """Load and standardize crisis data from various formats"""
        logger.info(f"📂 Loading {crisis_type} crisis data from {data_path}")
        
        all_data = []
        
        # Process all CSV files in the crisis directory
        for file_path in data_path.glob("*.csv"):
            try:
                logger.info(f"📄 Processing file: {file_path.name}")
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                
                # Standardize column names
                df = self._standardize_columns(df)
                
                # Add crisis type
                df['crisis_type'] = crisis_type
                df['source_file'] = file_path.name
                
                all_data.append(df)
                logger.info(f"✅ Loaded {len(df)} records from {file_path.name}")
                
            except Exception as e:
                logger.error(f"❌ Error loading {file_path.name}: {e}")
                continue
        
        if not all_data:
            logger.warning(f"⚠️ No data loaded for {crisis_type}")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"🔗 Combined {len(combined_df)} total records for {crisis_type}")
        
        return combined_df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different datasets"""
        
        # Common column mappings
        column_mappings = {
            # Text content
            'text': ['text', 'tweet', 'content', 'message', 'post'],
            'user_id': ['user_id', 'userid', 'user', 'username', 'author'],
            'timestamp': ['timestamp', 'created_at', 'date', 'time', 'datetime'],
            'id': ['id', 'tweet_id', 'post_id', 'message_id'],
            
            # Optional fields
            'retweet_count': ['retweet_count', 'retweets', 'rt_count'],
            'like_count': ['like_count', 'likes', 'favorite_count'],
            'location': ['location', 'place', 'geo', 'coordinates']
        }
        
        # Create standardized DataFrame
        standardized_df = pd.DataFrame()
        
        for standard_col, possible_cols in column_mappings.items():
            for col in possible_cols:
                if col in df.columns:
                    standardized_df[standard_col] = df[col]
                    break
            
            # If required column not found, create empty column
            if standard_col not in standardized_df.columns:
                if standard_col in ['text', 'user_id']:
                    logger.warning(f"⚠️ Required column '{standard_col}' not found")
                    standardized_df[standard_col] = None
                else:
                    standardized_df[standard_col] = None
        
        return standardized_df
    
    def clean_text(self, text: str, preserve_hashtags: bool = True, preserve_mentions: bool = False) -> str:
        """Clean and normalize text content"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.text_cleaning_patterns['urls'].sub('', text)
        
        # Handle hashtags and mentions
        if not preserve_hashtags:
            text = self.text_cleaning_patterns['hashtags'].sub('', text)
        else:
            # Keep hashtags but remove the # symbol
            text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
        
        if not preserve_mentions:
            text = self.text_cleaning_patterns['mentions'].sub('', text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra spaces
        text = self.text_cleaning_patterns['extra_spaces'].sub(' ', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def filter_data(self, df: pd.DataFrame, min_text_length: int = 10, 
                   max_text_length: int = 500) -> pd.DataFrame:
        """Filter data based on quality criteria"""
        logger.info(f"🔍 Filtering data: {len(df)} initial records")
        
        # Remove rows with missing essential data
        df = df.dropna(subset=['text', 'user_id'])
        logger.info(f"📝 After removing missing text/user_id: {len(df)} records")
        
        # Filter by text length
        df['text_length'] = df['text'].str.len()
        df = df[(df['text_length'] >= min_text_length) & (df['text_length'] <= max_text_length)]
        logger.info(f"📏 After text length filtering: {len(df)} records")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['text'], keep='first')
        logger.info(f"🔄 Removed {initial_count - len(df)} duplicate texts")
        
        # Remove bot-like patterns (excessive repetition)
        df = self._remove_bot_content(df)
        
        return df
    
    def _remove_bot_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove likely bot-generated content"""
        initial_count = len(df)
        
        # Remove users with excessive posting (likely bots)
        user_post_counts = df['user_id'].value_counts()
        bot_users = user_post_counts[user_post_counts > 100].index
        df = df[~df['user_id'].isin(bot_users)]
        
        logger.info(f"🤖 Removed {initial_count - len(df)} bot-like posts")
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for time-based analysis"""
        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            logger.warning("⚠️ No timestamp data available for temporal features")
            return df
        
        try:
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Extract temporal features
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['date'] = df['datetime'].dt.date
            
            logger.info("📅 Added temporal features")
            
        except Exception as e:
            logger.error(f"❌ Error adding temporal features: {e}")
        
        return df
    
    def prepare_for_analysis(self, crisis_type: str, data_path: Path, 
                           sample_size: Optional[int] = None) -> pd.DataFrame:
        """Complete preprocessing pipeline for crisis data"""
        logger.info(f"🚀 Starting preprocessing for {crisis_type} crisis")
        
        # Load data
        df = self.load_crisis_data(crisis_type, data_path)
        
        if df.empty:
            logger.error(f"❌ No data loaded for {crisis_type}")
            return df
        
        # Clean text
        df['text'] = df['text'].apply(lambda x: self.clean_text(x))
        
        # Filter data
        df = self.filter_data(df)
        
        # Add temporal features
        df = self.add_temporal_features(df)
        
        # Sample data if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"📊 Sampled {sample_size} records")
        
        # Add processing metadata
        df['processed_at'] = pd.Timestamp.now()
        
        logger.info(f"✅ Preprocessing complete: {len(df)} records ready for analysis")
        
        return df

if __name__ == "__main__":
    # Test preprocessing
    preprocessor = CrisisDataPreprocessor()
    
    # Test text cleaning
    test_text = "This is a TEST tweet with @mention #hashtag and https://example.com URL!!!"
    cleaned = preprocessor.clean_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    
    logger.info("✅ Preprocessor module tested successfully!")
