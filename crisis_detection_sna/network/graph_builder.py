"""
CrisisDetect+ Graph Builder
Constructs sentiment-emotion similarity networks for crisis analysis
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import itertools
from tqdm import tqdm

from utils.config import NETWORK_CONFIG
from utils.logger import get_logger

logger = get_logger("GraphBuilder")

class CrisisGraphBuilder:
    """Builds crisis-specific networks using sentiment-emotion similarity"""
    
    def __init__(self, similarity_threshold: float = None):
        self.similarity_threshold = similarity_threshold or NETWORK_CONFIG['similarity_threshold']
        self.min_nodes = NETWORK_CONFIG['min_nodes']
        self.max_nodes = NETWORK_CONFIG['max_nodes']
        self.edge_weight_method = NETWORK_CONFIG['edge_weight_method']
        
    def build_crisis_network(self, df: pd.DataFrame, crisis_type: str) -> nx.Graph:
        """Build network for specific crisis type using sentiment-emotion similarity"""
        logger.info(f"🌐 Building {crisis_type} crisis network from {len(df)} records")
        
        # Prepare user features
        user_features = self._prepare_user_features(df)
        
        if len(user_features) < self.min_nodes:
            logger.warning(f"⚠️ Insufficient users ({len(user_features)}) for network construction")
            return nx.Graph()
        
        # Limit users if too many
        if len(user_features) > self.max_nodes:
            logger.info(f"📊 Sampling {self.max_nodes} users from {len(user_features)} total")
            user_features = user_features.sample(n=self.max_nodes, random_state=42)
        
        # Build network
        G = self._construct_network(user_features, crisis_type)
        
        # Add network metadata
        G.graph['crisis_type'] = crisis_type
        G.graph['similarity_threshold'] = self.similarity_threshold
        G.graph['construction_method'] = 'sentiment_emotion_similarity'
        G.graph['total_records'] = len(df)
        G.graph['unique_users'] = len(user_features)
        
        logger.log_network_stats(crisis_type, G.number_of_nodes(), G.number_of_edges(), nx.density(G))
        
        return G
    
    def _prepare_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare user-level features for network construction"""
        
        # Required columns check
        required_cols = ['user_id', 'sentiment_compound']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Aggregate user features
        agg_dict = {
            'sentiment_compound': ['mean', 'std', 'count']
        }

        # Add emotion columns if they exist
        if 'emotion_crisis_intensity' in df.columns:
            agg_dict['emotion_crisis_intensity'] = 'mean'
        if 'emotion_primary' in df.columns:
            agg_dict['emotion_primary'] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'neutral'

        user_agg = df.groupby('user_id').agg(agg_dict).reset_index()
        
        # Flatten column names
        new_columns = ['user_id', 'sentiment_mean', 'sentiment_std', 'post_count']

        # Add emotion columns if they exist
        if 'emotion_crisis_intensity' in agg_dict:
            new_columns.append('emotion_intensity_mean')
        if 'emotion_primary' in agg_dict:
            new_columns.append('emotion_primary_mode')

        user_agg.columns = new_columns

        # Fill NaN values
        user_agg['sentiment_std'] = user_agg['sentiment_std'].fillna(0.0)
        if 'emotion_intensity_mean' in user_agg.columns:
            user_agg['emotion_intensity_mean'] = user_agg['emotion_intensity_mean'].fillna(0.0)
        if 'emotion_primary_mode' not in user_agg.columns:
            user_agg['emotion_primary_mode'] = 'neutral'
        
        # Add individual emotion scores if available
        emotion_cols = [col for col in df.columns if col.startswith('emotion_') and col not in ['emotion_primary', 'emotion_score', 'emotion_crisis_intensity', 'emotion_crisis_total']]
        
        for emotion_col in emotion_cols:
            emotion_name = emotion_col.replace('emotion_', '')
            user_agg[f'{emotion_name}_mean'] = df.groupby('user_id')[emotion_col].mean().values
        
        # Filter users with minimum activity
        min_posts = 2
        user_agg = user_agg[user_agg['post_count'] >= min_posts]
        
        logger.info(f"👥 Prepared features for {len(user_agg)} users")
        return user_agg
    
    def _construct_network(self, user_features: pd.DataFrame, crisis_type: str) -> nx.Graph:
        """Construct network using sentiment-emotion similarity"""
        
        G = nx.Graph()
        users = user_features['user_id'].tolist()
        
        # Add nodes with attributes
        for _, user_row in user_features.iterrows():
            user_id = user_row['user_id']
            node_attrs = {
                'sentiment_mean': user_row['sentiment_mean'],
                'sentiment_std': user_row['sentiment_std'],
                'post_count': user_row['post_count'],
                'emotion_intensity': user_row['emotion_intensity_mean'],
                'primary_emotion': user_row['emotion_primary_mode']
            }
            
            # Add emotion scores as node attributes
            for col in user_row.index:
                if col.endswith('_mean') and col not in ['sentiment_mean', 'emotion_intensity_mean']:
                    emotion_name = col.replace('_mean', '')
                    node_attrs[f'emotion_{emotion_name}'] = user_row[col]
            
            G.add_node(user_id, **node_attrs)
        
        # Calculate similarities and add edges
        edges_added = 0
        total_pairs = len(users) * (len(users) - 1) // 2
        
        logger.info(f"🔗 Calculating similarities for {total_pairs} user pairs")
        
        # Use batch processing for efficiency
        batch_size = 1000
        user_pairs = list(itertools.combinations(users, 2))
        
        for i in tqdm(range(0, len(user_pairs), batch_size), desc="Building edges"):
            batch_pairs = user_pairs[i:i + batch_size]
            
            for user1, user2 in batch_pairs:
                similarity = self._calculate_user_similarity(
                    user_features[user_features['user_id'] == user1].iloc[0],
                    user_features[user_features['user_id'] == user2].iloc[0]
                )
                
                if similarity >= self.similarity_threshold:
                    G.add_edge(user1, user2, weight=similarity, similarity=similarity)
                    edges_added += 1
        
        logger.info(f"✅ Added {edges_added} edges from {total_pairs} possible connections")
        
        return G
    
    def _calculate_user_similarity(self, user1: pd.Series, user2: pd.Series) -> float:
        """Calculate similarity between two users based on sentiment and emotions"""
        
        if self.edge_weight_method == 'sentiment_emotion_combined':
            return self._sentiment_emotion_similarity(user1, user2)
        elif self.edge_weight_method == 'sentiment_only':
            return self._sentiment_similarity(user1, user2)
        elif self.edge_weight_method == 'emotion_only':
            return self._emotion_similarity(user1, user2)
        else:
            return self._sentiment_emotion_similarity(user1, user2)
    
    def _sentiment_emotion_similarity(self, user1: pd.Series, user2: pd.Series) -> float:
        """Calculate combined sentiment-emotion similarity"""
        
        # Sentiment similarity (primary component)
        sentiment_sim = self._sentiment_similarity(user1, user2)
        
        # Emotion similarity
        emotion_sim = self._emotion_similarity(user1, user2)
        
        # Crisis intensity similarity
        intensity1 = user1.get('emotion_intensity_mean', 0.0)
        intensity2 = user2.get('emotion_intensity_mean', 0.0)
        intensity_sim = 1.0 - abs(intensity1 - intensity2)
        
        # Weighted combination
        combined_similarity = (
            0.5 * sentiment_sim +
            0.3 * emotion_sim +
            0.2 * intensity_sim
        )
        
        return max(0.0, min(1.0, combined_similarity))
    
    def _sentiment_similarity(self, user1: pd.Series, user2: pd.Series) -> float:
        """Calculate sentiment-based similarity"""
        
        sent1 = user1['sentiment_mean']
        sent2 = user2['sentiment_mean']
        
        # Sentiment distance similarity
        sentiment_distance = abs(sent1 - sent2)
        sentiment_sim = 1.0 - (sentiment_distance / 2.0)  # Normalize by max possible distance (2.0)
        
        return max(0.0, min(1.0, sentiment_sim))
    
    def _emotion_similarity(self, user1: pd.Series, user2: pd.Series) -> float:
        """Calculate emotion-based similarity using cosine similarity"""
        
        # Extract emotion features
        emotion_cols = [col for col in user1.index if col.endswith('_mean') and 'emotion' not in col.split('_')[0]]
        
        if not emotion_cols:
            return 0.5  # Default similarity if no emotion data
        
        # Create emotion vectors
        emotions1 = np.array([user1.get(col, 0.0) for col in emotion_cols])
        emotions2 = np.array([user2.get(col, 0.0) for col in emotion_cols])
        
        # Handle zero vectors
        if np.allclose(emotions1, 0) or np.allclose(emotions2, 0):
            return 0.5
        
        # Calculate cosine similarity
        try:
            similarity = cosine_similarity([emotions1], [emotions2])[0][0]
            return max(0.0, min(1.0, similarity))
        except:
            return 0.5
    
    def add_temporal_edges(self, G: nx.Graph, df: pd.DataFrame, time_window_hours: int = 24) -> nx.Graph:
        """Add temporal edges based on posting time proximity"""
        
        if 'datetime' not in df.columns:
            logger.warning("⚠️ No temporal data available for temporal edges")
            return G
        
        logger.info(f"⏰ Adding temporal edges with {time_window_hours}h window")
        
        # Group posts by user and time
        df_temporal = df[['user_id', 'datetime']].dropna()
        
        temporal_edges = 0
        for user1, user2 in G.edges():
            user1_times = df_temporal[df_temporal['user_id'] == user1]['datetime']
            user2_times = df_temporal[df_temporal['user_id'] == user2]['datetime']
            
            # Check for temporal proximity
            for t1 in user1_times:
                for t2 in user2_times:
                    time_diff = abs((t1 - t2).total_seconds() / 3600)  # Hours
                    if time_diff <= time_window_hours:
                        # Add temporal weight to existing edge
                        if G.has_edge(user1, user2):
                            current_weight = G[user1][user2].get('weight', 0.0)
                            temporal_bonus = 0.1 * (1.0 - time_diff / time_window_hours)
                            G[user1][user2]['weight'] = min(1.0, current_weight + temporal_bonus)
                            G[user1][user2]['temporal_proximity'] = True
                            temporal_edges += 1
                        break
                    
        logger.info(f"⏰ Enhanced {temporal_edges} edges with temporal proximity")
        return G
    
    def get_network_summary(self, G: nx.Graph) -> Dict:
        """Get comprehensive network summary"""
        
        if G.number_of_nodes() == 0:
            return {'error': 'Empty network'}
        
        summary = {
            'basic_stats': {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'is_connected': nx.is_connected(G),
                'components': nx.number_connected_components(G)
            },
            'weight_stats': {
                'avg_weight': np.mean([d['weight'] for u, v, d in G.edges(data=True) if 'weight' in d]),
                'min_weight': min([d['weight'] for u, v, d in G.edges(data=True) if 'weight' in d], default=0),
                'max_weight': max([d['weight'] for u, v, d in G.edges(data=True) if 'weight' in d], default=0)
            },
            'node_stats': {
                'avg_degree': np.mean([d for n, d in G.degree()]),
                'max_degree': max([d for n, d in G.degree()], default=0),
                'min_degree': min([d for n, d in G.degree()], default=0)
            }
        }
        
        return summary

if __name__ == "__main__":
    # Test graph builder
    builder = CrisisGraphBuilder()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(20)],
        'sentiment_compound': np.random.uniform(-1, 1, 20),
        'emotion_crisis_intensity': np.random.uniform(0, 1, 20),
        'emotion_primary': np.random.choice(['fear', 'anger', 'joy', 'neutral'], 20)
    })
    
    # Build network
    G = builder.build_crisis_network(sample_data, 'test')
    summary = builder.get_network_summary(G)
    
    print(f"Network Summary: {summary}")
    logger.info("✅ Graph builder tested successfully!")
