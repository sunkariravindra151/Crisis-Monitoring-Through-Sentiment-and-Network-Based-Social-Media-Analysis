"""
CrisisDetect+ Main Analysis System
Complete crisis analysis with 2500 records per crisis type
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(str(Path(__file__).parent))

from nlp.sentiment_classifier import CrisisSentimentClassifier
from nlp.emotion_classifier import CrisisEmotionClassifier
from network.graph_builder import CrisisGraphBuilder
from network.sna_metrics import CrisisSNAMetrics
from detection.crisis_score_calculator import CrisisScoreCalculator
from detection.alert_engine import CrisisAlertEngine
from utils.logger import CrisisLogger
from utils.config import CRISIS_TYPES

# Initialize logger
logger = CrisisLogger("CrisisDetect+")

class CrisisAnalyzer:
    """Main crisis analysis system"""
    
    def __init__(self):
        self.sample_size = 10000  # Fixed sample size per crisis
        self.crisis_types = ['health', 'technological', 'political', 'social']
        self.results = {}
        
        # Initialize components
        logger.info("Initializing CrisisDetect+ Analysis System...")
        self.sentiment_classifier = CrisisSentimentClassifier()
        self.emotion_classifier = CrisisEmotionClassifier()
        self.graph_builder = CrisisGraphBuilder()
        self.sna_metrics = CrisisSNAMetrics()
        self.crisis_calculator = CrisisScoreCalculator()
        self.alert_engine = CrisisAlertEngine()
        
        logger.info("All components initialized successfully!")
    
    def load_crisis_data(self, crisis_type):
        """Load and prepare data for a crisis type"""
        logger.info(f"Loading data for {crisis_type} crisis...")
        
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
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_data)} total records")
        
        # Sample exactly 2500 records
        if len(combined_data) > self.sample_size:
            combined_data = combined_data.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {self.sample_size} records for analysis")
        else:
            logger.info(f"Using all {len(combined_data)} available records")
        
        # Standardize columns
        combined_data = self._standardize_columns(combined_data)
        
        return combined_data
    
    def _standardize_columns(self, df):
        """Standardize column names"""
        # Standardize text column
        if 'text' not in df.columns:
            text_cols = ['tweet', 'content', 'message', 'post', 'full_text']
            for col in text_cols:
                if col in df.columns:
                    df['text'] = df[col]
                    break
            else:
                logger.warning("No text column found, using first string column")
                string_cols = df.select_dtypes(include=['object']).columns
                if len(string_cols) > 0:
                    df['text'] = df[string_cols[0]]
        
        # Standardize user_id column
        if 'user_id' not in df.columns:
            user_cols = ['user', 'username', 'author', 'userid', 'user_screen_name']
            for col in user_cols:
                if col in df.columns:
                    df['user_id'] = df[col]
                    break
            else:
                df['user_id'] = range(len(df))
        
        # Clean text data
        df['text'] = df['text'].astype(str).fillna('')
        df = df[df['text'].str.len() > 0]  # Remove empty texts
        
        return df
    
    def analyze_crisis(self, crisis_type, data):
        """Perform complete analysis for a crisis type"""
        logger.info(f"Starting analysis for {crisis_type} crisis...")
        
        # NLP Analysis
        logger.info("Performing NLP analysis...")
        texts = data['text'].tolist()
        
        # Sentiment analysis
        sentiment_results = self.sentiment_classifier.predict_batch(texts)
        sentiment_scores = [r['compound'] for r in sentiment_results]
        sentiment_labels = [r['label'] for r in sentiment_results]

        # Emotion analysis
        emotion_results = self.emotion_classifier.predict_batch(texts)
        emotion_scores = [r['emotion_scores'] for r in emotion_results]

        # Extract emotion metrics for network analysis
        emotion_primary = [r['primary_emotion'] for r in emotion_results]
        emotion_primary_scores = [r['primary_score'] for r in emotion_results]
        crisis_intensities = []

        for scores in emotion_scores:
            crisis_emotions = ['fear', 'anger', 'sadness']
            crisis_intensity = sum(scores.get(emotion, 0) for emotion in crisis_emotions)
            crisis_intensities.append(crisis_intensity)

        # Add results to dataframe
        data['sentiment_score'] = sentiment_scores
        data['sentiment_compound'] = sentiment_scores  # Graph builder expects this column
        data['sentiment_label'] = sentiment_labels
        data['emotion_primary'] = emotion_primary
        data['emotion_score'] = emotion_primary_scores
        data['emotion_crisis_intensity'] = crisis_intensities

        # Store emotion_scores separately for analysis (not in main DataFrame to avoid aggregation issues)
        emotion_data = emotion_scores
        
        # Calculate NLP metrics
        nlp_results = self._calculate_nlp_metrics(sentiment_scores, emotion_scores)
        logger.info(f"NLP analysis completed - Avg sentiment: {nlp_results['sentiment']['mean']:.3f}")
        
        # Network Analysis
        logger.info("Building crisis network...")
        network = self.graph_builder.build_crisis_network(data, crisis_type)
        
        # Calculate network metrics
        network_results = self._calculate_network_metrics(network, crisis_type)
        logger.info(f"Network analysis completed - {network_results['basic_stats']['nodes']} nodes, {network_results['basic_stats']['edges']} edges")
        
        # Crisis Detection
        logger.info("Calculating crisis severity...")
        crisis_result = self.crisis_calculator.calculate_crisis_severity(
            data, network, crisis_type, network_results
        )
        
        logger.info(f"Crisis severity calculated: {crisis_result['severity_score']:.3f} ({crisis_result['severity_level']})")
        
        # Generate alerts - combine all results for alert processing
        combined_results = {
            crisis_type: {
                'crisis_results': crisis_result,
                'nlp_results': nlp_results,
                'network_results': network_results
            }
        }
        alerts = self.alert_engine.process_crisis_results(combined_results)
        
        # Compile results
        analysis_result = {
            'crisis_type': crisis_type,
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_records': len(data),
                'unique_users': data['user_id'].nunique(),
                'analysis_period': 'complete_dataset'
            },
            'nlp_results': nlp_results,
            'network_results': network_results,
            'crisis_results': crisis_result,
            'alerts': alerts,
            'sample_data': data.head(10).to_dict('records')  # Sample for verification
        }
        
        return analysis_result
    
    def _calculate_nlp_metrics(self, sentiment_scores, emotion_scores):
        """Calculate NLP metrics"""
        sentiment_array = np.array(sentiment_scores)
        
        # Sentiment metrics
        sentiment_metrics = {
            'mean': float(np.mean(sentiment_array)),
            'std': float(np.std(sentiment_array)),
            'negative_ratio': float(np.sum(sentiment_array < -0.1) / len(sentiment_array)),
            'positive_ratio': float(np.sum(sentiment_array > 0.1) / len(sentiment_array)),
            'neutral_ratio': float(np.sum(np.abs(sentiment_array) <= 0.1) / len(sentiment_array))
        }
        
        # Emotion metrics
        crisis_emotions = ['fear', 'anger', 'sadness']
        crisis_intensities = []
        
        for scores in emotion_scores:
            crisis_intensity = sum(scores.get(emotion, 0) for emotion in crisis_emotions)
            crisis_intensities.append(crisis_intensity)
        
        emotion_metrics = {
            'mean_crisis_intensity': float(np.mean(crisis_intensities)),
            'std_crisis_intensity': float(np.std(crisis_intensities)),
            'high_intensity_ratio': float(np.sum(np.array(crisis_intensities) > 0.5) / len(crisis_intensities))
        }
        
        return {
            'sentiment': sentiment_metrics,
            'emotion': emotion_metrics
        }
    
    def _calculate_network_metrics(self, network, crisis_type):
        """Calculate network metrics"""
        basic_stats = {
            'nodes': network.number_of_nodes(),
            'edges': network.number_of_edges(),
            'density': float(network.number_of_edges() / max(1, network.number_of_nodes() * (network.number_of_nodes() - 1) / 2)) if network.number_of_nodes() > 1 else 0.0
        }
        
        centralities = self.sna_metrics.calculate_all_centralities(network, crisis_type)
        communities = self.sna_metrics.detect_communities(network, crisis_type)
        
        return {
            'basic_stats': basic_stats,
            'centralities': centralities,
            'communities': communities
        }
    
    def run_complete_analysis(self):
        """Run analysis for all crisis types"""
        logger.info("Starting complete crisis analysis...")
        logger.info(f"Analyzing {len(self.crisis_types)} crisis types with {self.sample_size} records each")
        
        all_results = {}
        crisis_rankings = []
        
        for crisis_type in self.crisis_types:
            try:
                # Load data
                data = self.load_crisis_data(crisis_type)
                if data is None:
                    logger.error(f"Skipping {crisis_type} - no data available")
                    continue
                
                # Analyze crisis
                result = self.analyze_crisis(crisis_type, data)
                all_results[crisis_type] = result
                
                # Add to rankings
                crisis_rankings.append({
                    'crisis_type': crisis_type,
                    'severity_score': result['crisis_results']['severity_score'],
                    'severity_level': result['crisis_results']['severity_level'],
                    'confidence': result['crisis_results']['confidence']
                })
                
                logger.info(f"Completed analysis for {crisis_type}")
                
            except Exception as e:
                logger.error(f"Failed to analyze {crisis_type}: {e}")
        
        # Sort rankings by severity
        crisis_rankings.sort(key=lambda x: x['severity_score'], reverse=True)
        
        # Generate summary
        summary = self._generate_summary(all_results, crisis_rankings)
        
        # Save results
        self._save_results(all_results, crisis_rankings, summary)
        
        # Print results
        self._print_results(crisis_rankings, summary)
        
        return all_results, crisis_rankings, summary

    def _generate_summary(self, all_results, crisis_rankings):
        """Generate analysis summary"""
        total_records = sum(result['data_summary']['total_records'] for result in all_results.values())
        total_users = sum(result['data_summary']['unique_users'] for result in all_results.values())

        # Count severity levels
        severity_counts = {'DETECTED': 0, 'WARNING': 0, 'MONITORED': 0, 'SAFE': 0}
        for ranking in crisis_rankings:
            level = ranking['severity_level']
            if level in severity_counts:
                severity_counts[level] += 1

        avg_severity = np.mean([r['severity_score'] for r in crisis_rankings]) if crisis_rankings else 0.0

        return {
            'total_crises_analyzed': len(all_results),
            'total_data_points': total_records,
            'total_unique_users': total_users,
            'avg_severity_score': float(avg_severity),
            'crises_detected': severity_counts['DETECTED'],
            'crises_warning': severity_counts['WARNING'],
            'crises_monitored': severity_counts['MONITORED'],
            'crises_safe': severity_counts['SAFE'],
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _save_results(self, all_results, crisis_rankings, summary):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save complete results
        complete_results = {
            'analysis_summary': summary,
            'crisis_rankings': crisis_rankings,
            'detailed_results': all_results,
            'metadata': {
                'analysis_type': 'complete_crisis_analysis',
                'sample_size_per_crisis': self.sample_size,
                'crisis_types_analyzed': list(all_results.keys()),
                'timestamp': timestamp
            }
        }

        results_file = output_dir / f"crisis_analysis_complete_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)

        # Save rankings CSV
        rankings_df = pd.DataFrame(crisis_rankings)
        rankings_file = output_dir / f"crisis_rankings_{timestamp}.csv"
        rankings_df.to_csv(rankings_file, index=False)

        # Save individual crisis results
        for crisis_type, result in all_results.items():
            crisis_file = output_dir / f"{crisis_type}_analysis_{timestamp}.json"
            with open(crisis_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Main results: {results_file.name}")
        logger.info(f"Rankings: {rankings_file.name}")

    def _print_results(self, crisis_rankings, summary):
        """Print analysis results to console"""
        print("\n" + "="*80)
        print("CRISISDETECT+ COMPLETE ANALYSIS RESULTS")
        print("="*80)

        # Summary
        print(f"ANALYSIS SUMMARY:")
        print(f"  Total Crises Analyzed: {summary['total_crises_analyzed']}")
        print(f"  Total Data Points: {summary['total_data_points']:,}")
        print(f"  Total Unique Users: {summary['total_unique_users']:,}")
        print(f"  Average Severity Score: {summary['avg_severity_score']:.3f}")

        # Risk breakdown
        print(f"\nRISK ASSESSMENT:")
        print(f"  [HIGH RISK] DETECTED: {summary['crises_detected']}")
        print(f"  [MEDIUM RISK] WARNING: {summary['crises_warning']}")
        print(f"  [LOW RISK] MONITORED: {summary['crises_monitored']}")
        print(f"  [SAFE] SAFE: {summary['crises_safe']}")

        # Rankings
        if crisis_rankings:
            print(f"\nCRISIS SEVERITY RANKINGS:")
            print("-" * 80)
            for i, crisis in enumerate(crisis_rankings, 1):
                risk_indicator = {
                    'DETECTED': '[HIGH RISK]',
                    'WARNING': '[MEDIUM RISK]',
                    'MONITORED': '[LOW RISK]',
                    'SAFE': '[SAFE]'
                }.get(crisis['severity_level'], '[UNKNOWN]')

                print(f"  {i}. {risk_indicator} {crisis['crisis_type'].title()} Crisis:")
                print(f"     Severity Score: {crisis['severity_score']:.3f}")
                print(f"     Risk Level: {crisis['severity_level']}")
                print(f"     Confidence: {crisis['confidence']:.3f}")
                print()

        # Key insights
        if crisis_rankings:
            highest = crisis_rankings[0]
            lowest = crisis_rankings[-1]

            print(f"KEY INSIGHTS:")
            print(f"  Most Critical: {highest['crisis_type'].title()} ({highest['severity_score']:.3f})")
            print(f"  Least Critical: {lowest['crisis_type'].title()} ({lowest['severity_score']:.3f})")

            high_risk_count = summary['crises_detected'] + summary['crises_warning']
            total_crises = summary['total_crises_analyzed']
            risk_percentage = (high_risk_count / total_crises) * 100 if total_crises > 0 else 0

            print(f"  Risk Distribution: {risk_percentage:.1f}% show elevated risk")

        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if summary['crises_detected'] > 0:
            print(f"  - Immediate attention required for high-risk crises")
            print(f"  - Activate crisis response protocols")
            print(f"  - Monitor situation closely")
        elif summary['crises_warning'] > 0:
            print(f"  - Enhanced monitoring for warning-level crises")
            print(f"  - Prepare response measures")
            print(f"  - Review crisis indicators")
        else:
            print(f"  - All crises within normal parameters")
            print(f"  - Continue routine monitoring")
            print(f"  - Maintain current protocols")

        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("Check results/outputs/ directory for detailed files.")
        print("="*80)


def main():
    """Main entry point"""
    print("="*80)
    print("CRISISDETECT+ COMPLETE CRISIS ANALYSIS")
    print("="*80)
    print("Configuration:")
    print("- Crisis Types: Health, Technological, Political, Social")
    print("- Sample Size: 10000 records per crisis type")
    print("- Total Records: ~40,000 records")
    print("- Analysis: Complete NLP + Network + Crisis Detection")
    print("="*80)

    try:
        # Initialize analyzer
        analyzer = CrisisAnalyzer()

        # Run complete analysis
        results, rankings, summary = analyzer.run_complete_analysis()

        print(f"\nAnalysis completed successfully!")
        print(f"Analyzed {len(results)} crisis types with {summary['total_data_points']:,} total records")

        return True

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Analysis failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
