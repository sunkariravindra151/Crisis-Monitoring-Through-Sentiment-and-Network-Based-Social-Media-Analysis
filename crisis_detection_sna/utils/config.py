"""
CrisisDetect+ Configuration
Central configuration for all system parameters
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"

# Crisis Types Configuration
CRISIS_TYPES = {
    'health': {
        'name': 'Health Crisis',
        'data_path': DATA_DIR / 'health',
        'data_files': [
            {'filename': 'Covid-19 Twitter Dataset (Apr-Jun 2021).csv'},
            {'filename': 'covid19_tweets.csv'}
        ],
        'keywords': ['covid', 'pandemic', 'virus', 'disease', 'health', 'hospital', 'vaccine'],
        'severity_weights': {
            'sentiment': 0.35,
            'emotion': 0.25,
            'volume': 0.25,
            'network': 0.15
        }
    },
    'technological': {
        'name': 'Technological Crisis',
        'data_path': DATA_DIR / 'technological',
        'data_files': [
            {'filename': 'Facebook_outage_Tweet_data_4th_October.csv'}
        ],
        'keywords': ['outage', 'down', 'server', 'crash', 'bug', 'error', 'failure'],
        'severity_weights': {
            'sentiment': 0.30,
            'emotion': 0.20,
            'volume': 0.35,
            'network': 0.15
        }
    },
    'political': {
        'name': 'Political Crisis',
        'data_path': DATA_DIR / 'political',
        'data_files': [
            {'filename': 'Capital -riots.csv'},
            {'filename': 'hashtag_donaldtrump.csv'}
        ],
        'keywords': ['election', 'vote', 'politics', 'government', 'protest', 'riot'],
        'severity_weights': {
            'sentiment': 0.25,
            'emotion': 0.35,
            'volume': 0.25,
            'network': 0.15
        }
    },
    'social': {
        'name': 'Social Crisis',
        'data_path': DATA_DIR / 'social',
        'data_files': [
            {'filename': 'IRAhandle_tweets_1.csv'}
        ],
        'keywords': ['social', 'community', 'society', 'people', 'public', 'crowd'],
        'severity_weights': {
            'sentiment': 0.30,
            'emotion': 0.30,
            'volume': 0.20,
            'network': 0.20
        }
    }
}

# NLP Configuration
NLP_CONFIG = {
    'sentiment_model': 'distilbert-base-uncased-finetuned-sst-2-english',
    'emotion_model': 'j-hartmann/emotion-english-distilroberta-base',
    'max_length': 512,
    'batch_size': 32,
    'device': 'auto'  # Will auto-detect GPU/CPU
}

# Network Configuration
NETWORK_CONFIG = {
    'similarity_threshold': 0.7,
    'min_nodes': 10,
    'max_nodes': 1000,
    'edge_weight_method': 'sentiment_emotion_combined',
    'centrality_measures': ['degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank']
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'sample_size': 2500,  # Fixed sample size per crisis type
    'enable_sampling': True,  # Enable sampling for consistent analysis
    'random_seed': 42  # For reproducible sampling
}

# Crisis Detection Thresholds
CRISIS_THRESHOLDS = {
    'safe': 0.0,
    'monitored': 0.15,
    'warning': 0.25,
    'detected': 0.4
}

# Visualization Configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
    'save_format': 'png'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': PROJECT_ROOT / 'logs' / 'crisisdetect.log'
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'max_workers': 4,
    'memory_limit_gb': 8,
    'enable_multiprocessing': True,
    'cache_results': True
}

def get_crisis_config(crisis_type: str) -> dict:
    """Get configuration for specific crisis type"""
    return CRISIS_TYPES.get(crisis_type, {})

def get_data_path(crisis_type: str) -> Path:
    """Get data path for specific crisis type"""
    config = get_crisis_config(crisis_type)
    return config.get('data_path', DATA_DIR)

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        DATA_DIR,
        RESULTS_DIR,
        OUTPUTS_DIR,
        VISUALIZATIONS_DIR,
        PROJECT_ROOT / 'logs'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("✅ CrisisDetect+ configuration loaded successfully!")
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"📊 Crisis types: {list(CRISIS_TYPES.keys())}")
    print(f"🤖 NLP models: {NLP_CONFIG['sentiment_model']}, {NLP_CONFIG['emotion_model']}")
