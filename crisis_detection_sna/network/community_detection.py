"""
CrisisDetect+ Community Detection
Advanced community detection and analysis for crisis networks
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set
import community as community_louvain
from collections import defaultdict, Counter
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger

logger = get_logger("CommunityDetection")

class CrisisCommunityDetector:
    """Advanced community detection for crisis networks"""
    
    def __init__(self):
        self.community_cache = {}
        
    def detect_communities_multi_method(self, G: nx.Graph, crisis_type: str) -> Dict[str, Dict]:
        """Detect communities using multiple methods and compare results"""
        logger.info(f"👥 Running multi-method community detection for {crisis_type}")
        
        if G.number_of_nodes() == 0:
            return {'error': 'Empty network'}
        
        results = {}
        
        # Method 1: Louvain Algorithm
        results['louvain'] = self._louvain_detection(G)
        
        # Method 2: Greedy Modularity
        results['greedy_modularity'] = self._greedy_modularity_detection(G)
        
        # Method 3: Label Propagation
        results['label_propagation'] = self._label_propagation_detection(G)
        
        # Method 4: Leiden Algorithm (if available)
        try:
            results['leiden'] = self._leiden_detection(G)
        except ImportError:
            logger.info("📝 Leiden algorithm not available (requires leidenalg package)")
        
        # Compare methods and select best
        best_method = self._select_best_method(results)
        results['best_method'] = best_method
        results['recommended'] = results.get(best_method, results['louvain'])
        
        # Cache results
        self.community_cache[crisis_type] = results
        
        logger.info(f"✅ Community detection complete. Best method: {best_method}")
        
        return results
    
    def _louvain_detection(self, G: nx.Graph) -> Dict:
        """Louvain community detection"""
        try:
            partition = community_louvain.best_partition(G, weight='weight', random_state=42)
            modularity = community_louvain.modularity(partition, G, weight='weight')
            
            communities = self._partition_to_communities(partition)
            
            return {
                'method': 'louvain',
                'partition': partition,
                'communities': communities,
                'modularity': modularity,
                'num_communities': len(communities),
                'quality_score': modularity
            }
        except Exception as e:
            logger.error(f"❌ Louvain detection failed: {e}")
            return {'error': str(e)}
    
    def _greedy_modularity_detection(self, G: nx.Graph) -> Dict:
        """Greedy modularity maximization"""
        try:
            communities_generator = nx.community.greedy_modularity_communities(G, weight='weight')
            communities_list = list(communities_generator)
            
            # Convert to partition format
            partition = {}
            for i, community in enumerate(communities_list):
                for node in community:
                    partition[node] = i
            
            modularity = nx.community.modularity(G, communities_list, weight='weight')
            
            communities = {i: list(comm) for i, comm in enumerate(communities_list)}
            
            return {
                'method': 'greedy_modularity',
                'partition': partition,
                'communities': communities,
                'modularity': modularity,
                'num_communities': len(communities_list),
                'quality_score': modularity
            }
        except Exception as e:
            logger.error(f"❌ Greedy modularity detection failed: {e}")
            return {'error': str(e)}
    
    def _label_propagation_detection(self, G: nx.Graph) -> Dict:
        """Label propagation algorithm"""
        try:
            communities_generator = nx.community.label_propagation_communities(G, weight='weight', seed=42)
            communities_list = list(communities_generator)
            
            # Convert to partition format
            partition = {}
            for i, community in enumerate(communities_list):
                for node in community:
                    partition[node] = i
            
            modularity = nx.community.modularity(G, communities_list, weight='weight')
            
            communities = {i: list(comm) for i, comm in enumerate(communities_list)}
            
            return {
                'method': 'label_propagation',
                'partition': partition,
                'communities': communities,
                'modularity': modularity,
                'num_communities': len(communities_list),
                'quality_score': modularity
            }
        except Exception as e:
            logger.error(f"❌ Label propagation detection failed: {e}")
            return {'error': str(e)}
    
    def _leiden_detection(self, G: nx.Graph) -> Dict:
        """Leiden algorithm (requires leidenalg package)"""
        try:
            import leidenalg
            import igraph as ig
            
            # Convert NetworkX to igraph
            edge_list = [(u, v, d.get('weight', 1.0)) for u, v, d in G.edges(data=True)]
            ig_graph = ig.Graph.TupleList(edge_list, weights=True)
            
            # Run Leiden algorithm
            partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition, weights='weight', seed=42)
            
            # Convert back to NetworkX format
            node_mapping = {i: list(G.nodes())[i] for i in range(len(G.nodes()))}
            nx_partition = {}
            communities = {}
            
            for comm_id, community in enumerate(partition):
                communities[comm_id] = []
                for node_idx in community:
                    node = node_mapping[node_idx]
                    nx_partition[node] = comm_id
                    communities[comm_id].append(node)
            
            modularity = partition.modularity
            
            return {
                'method': 'leiden',
                'partition': nx_partition,
                'communities': communities,
                'modularity': modularity,
                'num_communities': len(communities),
                'quality_score': modularity
            }
        except ImportError:
            raise ImportError("Leiden algorithm requires 'leidenalg' and 'python-igraph' packages")
        except Exception as e:
            logger.error(f"❌ Leiden detection failed: {e}")
            return {'error': str(e)}
    
    def _partition_to_communities(self, partition: Dict[str, int]) -> Dict[int, List[str]]:
        """Convert partition to communities format"""
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        return dict(communities)
    
    def _select_best_method(self, results: Dict[str, Dict]) -> str:
        """Select best community detection method based on quality metrics"""
        
        valid_results = {method: result for method, result in results.items() 
                        if 'error' not in result and 'quality_score' in result}
        
        if not valid_results:
            return 'louvain'  # Default fallback
        
        # Score each method
        method_scores = {}
        
        for method, result in valid_results.items():
            score = 0.0
            
            # Modularity score (primary metric)
            modularity = result.get('modularity', 0.0)
            score += modularity * 0.6
            
            # Number of communities (prefer reasonable number)
            num_communities = result.get('num_communities', 1)
            total_nodes = sum(len(comm) for comm in result.get('communities', {}).values())
            
            if total_nodes > 0:
                community_ratio = num_communities / total_nodes
                # Prefer 10-30% of nodes as communities
                if 0.1 <= community_ratio <= 0.3:
                    score += 0.2
                elif 0.05 <= community_ratio <= 0.5:
                    score += 0.1
            
            # Community size distribution (prefer balanced communities)
            communities = result.get('communities', {})
            if communities:
                sizes = [len(comm) for comm in communities.values()]
                size_std = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 1.0
                balance_score = max(0, 1.0 - size_std)
                score += balance_score * 0.2
            
            method_scores[method] = score
        
        # Return method with highest score
        best_method = max(method_scores.keys(), key=lambda k: method_scores[k])
        
        logger.info(f"📊 Method scores: {method_scores}")
        logger.info(f"🏆 Best method: {best_method} (score: {method_scores[best_method]:.3f})")
        
        return best_method
    
    def analyze_community_characteristics(self, G: nx.Graph, communities: Dict[int, List[str]], 
                                        crisis_type: str) -> Dict[str, Dict]:
        """Analyze characteristics of detected communities"""
        logger.info(f"🔍 Analyzing community characteristics for {crisis_type}")
        
        community_analysis = {}
        
        for comm_id, members in communities.items():
            if len(members) < 2:
                continue
                
            # Create subgraph for community
            subgraph = G.subgraph(members)
            
            # Basic statistics
            analysis = {
                'size': len(members),
                'density': nx.density(subgraph),
                'clustering': nx.average_clustering(subgraph, weight='weight'),
                'members': members
            }
            
            # Sentiment characteristics
            sentiment_stats = self._analyze_community_sentiment(G, members)
            analysis.update(sentiment_stats)
            
            # Network position
            position_stats = self._analyze_community_position(G, members)
            analysis.update(position_stats)
            
            # Crisis relevance
            crisis_stats = self._analyze_community_crisis_relevance(G, members)
            analysis.update(crisis_stats)
            
            community_analysis[comm_id] = analysis
        
        # Add inter-community analysis
        inter_community_stats = self._analyze_inter_community_connections(G, communities)
        
        return {
            'communities': community_analysis,
            'inter_community': inter_community_stats,
            'summary': self._summarize_community_analysis(community_analysis)
        }
    
    def _analyze_community_sentiment(self, G: nx.Graph, members: List[str]) -> Dict[str, float]:
        """Analyze sentiment characteristics of community members"""
        
        sentiments = []
        emotions = []
        
        for member in members:
            node_data = G.nodes[member]
            if 'sentiment_mean' in node_data:
                sentiments.append(node_data['sentiment_mean'])
            if 'emotion_intensity' in node_data:
                emotions.append(node_data['emotion_intensity'])
        
        stats = {}
        
        if sentiments:
            stats['avg_sentiment'] = np.mean(sentiments)
            stats['sentiment_std'] = np.std(sentiments)
            stats['sentiment_homogeneity'] = 1.0 - (np.std(sentiments) / 2.0)  # Normalized by max possible std
        
        if emotions:
            stats['avg_emotion_intensity'] = np.mean(emotions)
            stats['emotion_std'] = np.std(emotions)
        
        return stats
    
    def _analyze_community_position(self, G: nx.Graph, members: List[str]) -> Dict[str, float]:
        """Analyze network position of community"""
        
        # Internal vs external connections
        internal_edges = 0
        external_edges = 0
        
        for member in members:
            for neighbor in G.neighbors(member):
                if neighbor in members:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        # Avoid double counting internal edges
        internal_edges = internal_edges // 2
        
        total_edges = internal_edges + external_edges
        
        stats = {
            'internal_edges': internal_edges,
            'external_edges': external_edges,
            'internal_ratio': internal_edges / total_edges if total_edges > 0 else 0.0,
            'external_ratio': external_edges / total_edges if total_edges > 0 else 0.0
        }
        
        return stats
    
    def _analyze_community_crisis_relevance(self, G: nx.Graph, members: List[str]) -> Dict[str, float]:
        """Analyze crisis relevance of community"""
        
        crisis_indicators = []
        post_counts = []
        
        for member in members:
            node_data = G.nodes[member]
            
            # Crisis intensity from emotions
            if 'emotion_intensity' in node_data:
                crisis_indicators.append(node_data['emotion_intensity'])
            
            # Activity level
            if 'post_count' in node_data:
                post_counts.append(node_data['post_count'])
        
        stats = {}
        
        if crisis_indicators:
            stats['avg_crisis_intensity'] = np.mean(crisis_indicators)
            stats['crisis_intensity_std'] = np.std(crisis_indicators)
        
        if post_counts:
            stats['avg_activity'] = np.mean(post_counts)
            stats['total_activity'] = sum(post_counts)
        
        return stats
    
    def _analyze_inter_community_connections(self, G: nx.Graph, communities: Dict[int, List[str]]) -> Dict:
        """Analyze connections between communities"""
        
        inter_connections = defaultdict(int)
        community_lookup = {}
        
        # Create reverse lookup
        for comm_id, members in communities.items():
            for member in members:
                community_lookup[member] = comm_id
        
        # Count inter-community edges
        for u, v in G.edges():
            comm_u = community_lookup.get(u)
            comm_v = community_lookup.get(v)
            
            if comm_u is not None and comm_v is not None and comm_u != comm_v:
                # Sort to avoid double counting
                edge_key = tuple(sorted([comm_u, comm_v]))
                inter_connections[edge_key] += 1
        
        # Calculate statistics
        total_inter_edges = sum(inter_connections.values())
        total_edges = G.number_of_edges()
        
        stats = {
            'total_inter_community_edges': total_inter_edges,
            'inter_community_ratio': total_inter_edges / total_edges if total_edges > 0 else 0.0,
            'community_connections': dict(inter_connections),
            'most_connected_communities': sorted(inter_connections.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        return stats
    
    def _summarize_community_analysis(self, community_analysis: Dict[str, Dict]) -> Dict[str, float]:
        """Summarize community analysis results"""
        
        if not community_analysis:
            return {}
        
        sizes = [comm['size'] for comm in community_analysis.values()]
        densities = [comm['density'] for comm in community_analysis.values()]
        sentiments = [comm.get('avg_sentiment', 0.0) for comm in community_analysis.values()]
        
        summary = {
            'num_communities': len(community_analysis),
            'avg_community_size': np.mean(sizes),
            'largest_community': max(sizes),
            'smallest_community': min(sizes),
            'avg_community_density': np.mean(densities),
            'avg_community_sentiment': np.mean(sentiments),
            'sentiment_diversity': np.std(sentiments)
        }
        
        return summary

if __name__ == "__main__":
    # Test community detection
    detector = CrisisCommunityDetector()
    
    # Create sample network with communities
    G = nx.karate_club_graph()
    
    # Add random attributes
    for node in G.nodes():
        G.nodes[node]['sentiment_mean'] = np.random.uniform(-1, 1)
        G.nodes[node]['emotion_intensity'] = np.random.uniform(0, 1)
        G.nodes[node]['post_count'] = np.random.randint(1, 20)
    
    # Add edge weights
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
    
    # Detect communities
    results = detector.detect_communities_multi_method(G, 'test')
    
    if 'recommended' in results:
        communities = results['recommended'].get('communities', {})
        analysis = detector.analyze_community_characteristics(G, communities, 'test')
        
        print(f"Detected {len(communities)} communities")
        print(f"Best method: {results['best_method']}")
        print(f"Modularity: {results['recommended'].get('modularity', 0.0):.3f}")
    
    logger.info("✅ Community detection tested successfully!")
