"""
CrisisDetect+ Test System
Simple test to verify all components are working
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(str(Path(__file__).parent))

from nlp.sentiment_classifier import CrisisSentimentClassifier
from nlp.emotion_classifier import CrisisEmotionClassifier
from network.graph_builder import CrisisGraphBuilder
from network.sna_metrics import CrisisSNAMetrics
from detection.crisis_score_calculator import CrisisScoreCalculator
from utils.logger import CrisisLogger
from utils.config import CRISIS_TYPES

# Initialize logger
logger = CrisisLogger("CrisisTest")

def load_sample_data(crisis_type, sample_size=100):
    """Load sample data for testing"""
    logger.info(f"Loading sample data for {crisis_type} crisis...")
    
    data_files = CRISIS_TYPES[crisis_type]['data_files']
    all_data = []
    
    for file_info in data_files:
        file_path = Path("data") / crisis_type / file_info['filename']
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                logger.info(f"Loaded {len(df)} records from {file_info['filename']}")
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_info['filename']}: {e}")
    
    if not all_data:
        logger.error(f"No data files found for {crisis_type}")
        return None
    
    # Combine and sample data
    combined_data = pd.concat(all_data, ignore_index=True)
    if len(combined_data) > sample_size:
        combined_data = combined_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Standardize columns
    if 'text' not in combined_data.columns:
        text_cols = ['tweet', 'content', 'message', 'post']
        for col in text_cols:
            if col in combined_data.columns:
                combined_data['text'] = combined_data[col]
                break
    
    if 'user_id' not in combined_data.columns:
        user_cols = ['user', 'username', 'author', 'userid']
        for col in user_cols:
            if col in combined_data.columns:
                combined_data['user_id'] = combined_data[col]
                break
        else:
            combined_data['user_id'] = range(len(combined_data))
    
    logger.info(f"Prepared {len(combined_data)} records for {crisis_type} testing")
    return combined_data

def test_nlp_components():
    """Test NLP components"""
    print("\n" + "="*60)
    print("TESTING NLP COMPONENTS")
    print("="*60)
    
    try:
        # Test sentiment classifier
        print("Testing Sentiment Classifier...")
        sentiment_classifier = CrisisSentimentClassifier()
        
        test_texts = [
            "This health crisis is really concerning and scary",
            "Great technological advancement, very positive news",
            "Political situation is getting worse every day"
        ]
        
        results = sentiment_classifier.predict_batch(test_texts)
        for i, (text, result) in enumerate(zip(test_texts, results)):
            print(f"  Text {i+1}: {result['label']} ({result['score']:.3f})")

        print("[OK] Sentiment Classifier working!")

        # Test emotion classifier
        print("\nTesting Emotion Classifier...")
        emotion_classifier = CrisisEmotionClassifier()

        emotion_results = emotion_classifier.predict_batch(test_texts)
        for i, (text, result) in enumerate(zip(test_texts, emotion_results)):
            print(f"  Text {i+1}: {result['primary_emotion']} ({result['primary_score']:.3f})")
        
        print("[OK] Emotion Classifier working!")
        return True
        
    except Exception as e:
        print(f"[ERROR] NLP test failed: {e}")
        return False

def test_network_components():
    """Test network components"""
    print("\n" + "="*60)
    print("TESTING NETWORK COMPONENTS")
    print("="*60)
    
    try:
        # Create sample data with proper format
        sample_data = pd.DataFrame({
            'text': [f"Sample crisis text {i}" for i in range(20)],
            'user_id': [f"user_{i%5}" for i in range(20)],  # 5 unique users
            'sentiment_compound': np.random.uniform(-1, 1, 20),
            'sentiment_score': np.random.uniform(-1, 1, 20),
            'emotion_primary': ['fear', 'anger', 'sadness', 'joy', 'neutral'] * 4,
            'emotion_score': np.random.uniform(0, 1, 20),
            'emotion_crisis_intensity': np.random.uniform(0, 1, 20)
        })
        
        # Test graph builder
        print("Testing Graph Builder...")
        graph_builder = CrisisGraphBuilder()
        network = graph_builder.build_crisis_network(sample_data, "test")

        print(f"  Network created: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")

        # Test SNA metrics
        print("Testing SNA Metrics...")
        sna_metrics = CrisisSNAMetrics()
        metrics = sna_metrics.calculate_all_centralities(network, "test")
        
        print(f"  Centralities calculated for {len(metrics)} nodes")
        print("[OK] Network components working!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Network test failed: {e}")
        return False

def test_crisis_detection():
    """Test crisis detection"""
    print("\n" + "="*60)
    print("TESTING CRISIS DETECTION")
    print("="*60)
    
    try:
        # Load sample health data
        sample_data = load_sample_data('health', 50)
        if sample_data is None:
            print("[ERROR] Could not load sample data")
            return False
        
        # Test crisis score calculator
        print("Testing Crisis Score Calculator...")
        calculator = CrisisScoreCalculator()
        
        # Create mock analysis results
        nlp_results = {
            'sentiment': {
                'mean': -0.3,
                'std': 0.5,
                'negative_ratio': 0.7,
                'positive_ratio': 0.2
            },
            'emotion': {
                'mean_crisis_intensity': 0.4,
                'high_intensity_ratio': 0.3
            }
        }
        
        network_results = {
            'basic_stats': {'nodes': 10, 'edges': 15, 'density': 0.3},
            'centralities': {}
        }
        
        # Create a simple network for testing
        import networkx as nx
        test_network = nx.Graph()
        test_network.add_nodes_from(range(5))
        test_network.add_edges_from([(0,1), (1,2), (2,3), (3,4)])

        crisis_result = calculator.calculate_crisis_severity(
            sample_data, test_network, 'health', network_results
        )
        
        print(f"  Crisis Severity: {crisis_result['severity_score']:.3f}")
        print(f"  Risk Level: {crisis_result['severity_level']}")
        print(f"  Confidence: {crisis_result['confidence']:.3f}")
        
        print("[OK] Crisis Detection working!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Crisis detection test failed: {e}")
        return False

def run_full_test():
    """Run complete system test"""
    print("="*60)
    print("CRISISDETECT+ SYSTEM TEST")
    print("="*60)
    print("Testing all components with sample data...")
    
    tests = [
        ("NLP Components", test_nlp_components),
        ("Network Components", test_network_components),
        ("Crisis Detection", test_crisis_detection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n[PASS] {test_name}")
            else:
                print(f"\n[FAIL] {test_name}")
        except Exception as e:
            print(f"\n[FAIL] {test_name}: {e}")
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total-passed}/{total}")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! System is ready.")
        print("\nTo run full analysis:")
        print("  python main.py")
    else:
        print("\n[WARNING] Some tests failed. Check components.")
    
    return passed == total

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)
