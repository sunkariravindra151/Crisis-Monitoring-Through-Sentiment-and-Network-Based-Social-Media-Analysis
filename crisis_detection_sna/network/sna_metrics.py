"""
CrisisDetect+ SNA Metrics Calculator
Comprehensive social network analysis metrics for crisis networks
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import community as community_louvain
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from utils.config import NETWORK_CONFIG
from utils.logger import get_logger

logger = get_logger("SNAMetrics")

class CrisisSNAMetrics:
    """Calculate comprehensive SNA metrics for crisis networks"""
    
    def __init__(self):
        self.centrality_measures = NETWORK_CONFIG['centrality_measures']
        self.centrality_cache = {}
        
    def calculate_all_centralities(self, G: nx.Graph, crisis_type: str) -> Dict[str, Dict[str, float]]:
        """Calculate all centrality measures for the network"""
        logger.info(f"📊 Calculating centrality measures for {crisis_type} network")
        
        if G.number_of_nodes() == 0:
            logger.warning("⚠️ Empty network - cannot calculate centralities")
            return {}
        
        centralities = {}
        
        # Degree Centrality
        if 'degree' in self.centrality_measures:
            logger.info("🔢 Calculating degree centrality")
            centralities['degree'] = nx.degree_centrality(G)
        
        # Betweenness Centrality
        if 'betweenness' in self.centrality_measures:
            logger.info("🌉 Calculating betweenness centrality")
            try:
                centralities['betweenness'] = nx.betweenness_centrality(G, normalized=True)
            except Exception as e:
                logger.warning(f"⚠️ Betweenness calculation failed: {e}")
                centralities['betweenness'] = {node: 0.0 for node in G.nodes()}
        
        # Closeness Centrality
        if 'closeness' in self.centrality_measures:
            logger.info("📍 Calculating closeness centrality")
            try:
                centralities['closeness'] = nx.closeness_centrality(G)
            except Exception as e:
                logger.warning(f"⚠️ Closeness calculation failed: {e}")
                centralities['closeness'] = {node: 0.0 for node in G.nodes()}
        
        # Eigenvector Centrality
        if 'eigenvector' in self.centrality_measures:
            logger.info("🎯 Calculating eigenvector centrality")
            try:
                centralities['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
            except Exception as e:
                logger.warning(f"⚠️ Eigenvector calculation failed: {e}")
                centralities['eigenvector'] = {node: 0.0 for node in G.nodes()}
        
        # PageRank
        if 'pagerank' in self.centrality_measures:
            logger.info("🔍 Calculating PageRank")
            try:
                centralities['pagerank'] = nx.pagerank(G, alpha=0.85, max_iter=1000, tol=1e-06)
            except Exception as e:
                logger.warning(f"⚠️ PageRank calculation failed: {e}")
                centralities['pagerank'] = {node: 0.0 for node in G.nodes()}
        
        # Cache results
        self.centrality_cache[crisis_type] = centralities
        
        # Log statistics
        self._log_centrality_stats(centralities, crisis_type)
        
        return centralities
    
    def _log_centrality_stats(self, centralities: Dict[str, Dict], crisis_type: str):
        """Log centrality statistics"""
        for measure, values in centralities.items():
            if values:
                stats = {
                    'mean': np.mean(list(values.values())),
                    'std': np.std(list(values.values())),
                    'max': max(values.values()),
                    'min': min(values.values())
                }
                logger.info(f"📈 {measure.upper()} [{crisis_type}] - Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, Max: {stats['max']:.4f}")
    
    def detect_communities(self, G: nx.Graph, crisis_type: str) -> Dict[str, Union[Dict, float, int]]:
        """Detect communities using Louvain algorithm"""
        logger.info(f"👥 Detecting communities in {crisis_type} network")
        
        if G.number_of_nodes() == 0:
            return {'error': 'Empty network'}
        
        try:
            # Louvain community detection
            partition = community_louvain.best_partition(G, weight='weight', random_state=42)
            
            # Calculate modularity
            modularity = community_louvain.modularity(partition, G, weight='weight')
            
            # Organize communities
            communities = defaultdict(list)
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
            
            # Community statistics
            community_sizes = [len(members) for members in communities.values()]
            
            community_info = {
                'partition': partition,
                'communities': dict(communities),
                'modularity': modularity,
                'num_communities': len(communities),
                'community_sizes': community_sizes,
                'avg_community_size': np.mean(community_sizes),
                'largest_community_size': max(community_sizes) if community_sizes else 0,
                'smallest_community_size': min(community_sizes) if community_sizes else 0
            }
            
            logger.info(f"👥 Found {len(communities)} communities with modularity {modularity:.4f}")
            
            return community_info
            
        except Exception as e:
            logger.error(f"❌ Community detection failed: {e}")
            return {'error': str(e)}
    
    def calculate_network_cohesion(self, G: nx.Graph, crisis_type: str) -> Dict[str, float]:
        """Calculate network cohesion metrics"""
        logger.info(f"🔗 Calculating network cohesion for {crisis_type}")
        
        if G.number_of_nodes() == 0:
            return {'error': 'Empty network'}
        
        cohesion_metrics = {}
        
        try:
            # Basic cohesion metrics
            cohesion_metrics['density'] = nx.density(G)
            cohesion_metrics['transitivity'] = nx.transitivity(G)
            cohesion_metrics['average_clustering'] = nx.average_clustering(G, weight='weight')
            
            # Connectivity metrics
            cohesion_metrics['is_connected'] = nx.is_connected(G)
            cohesion_metrics['number_of_components'] = nx.number_connected_components(G)
            
            # Path-based metrics (for connected components)
            if nx.is_connected(G):
                cohesion_metrics['average_shortest_path'] = nx.average_shortest_path_length(G, weight='weight')
                cohesion_metrics['diameter'] = nx.diameter(G)
                cohesion_metrics['radius'] = nx.radius(G)
            else:
                # Calculate for largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                
                cohesion_metrics['average_shortest_path_lcc'] = nx.average_shortest_path_length(subgraph, weight='weight')
                cohesion_metrics['diameter_lcc'] = nx.diameter(subgraph)
                cohesion_metrics['radius_lcc'] = nx.radius(subgraph)
                cohesion_metrics['lcc_size'] = len(largest_cc)
                cohesion_metrics['lcc_ratio'] = len(largest_cc) / G.number_of_nodes()
            
            # Edge weight statistics
            if G.number_of_edges() > 0:
                weights = [d['weight'] for u, v, d in G.edges(data=True) if 'weight' in d]
                if weights:
                    cohesion_metrics['avg_edge_weight'] = np.mean(weights)
                    cohesion_metrics['edge_weight_std'] = np.std(weights)
                    cohesion_metrics['min_edge_weight'] = min(weights)
                    cohesion_metrics['max_edge_weight'] = max(weights)
            
            logger.info(f"🔗 Cohesion calculated - Density: {cohesion_metrics['density']:.4f}, Clustering: {cohesion_metrics['average_clustering']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Cohesion calculation failed: {e}")
            cohesion_metrics['error'] = str(e)
        
        return cohesion_metrics
    
    def identify_influencers(self, centralities: Dict[str, Dict], top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Identify top influencers based on centrality measures"""
        logger.info(f"🌟 Identifying top {top_k} influencers")
        
        influencers = {}
        
        for measure, values in centralities.items():
            if values:
                # Sort by centrality value and get top k
                sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)
                influencers[measure] = sorted_nodes[:top_k]
        
        # Calculate combined influence score
        if len(centralities) > 1:
            combined_scores = self._calculate_combined_influence(centralities)
            sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            influencers['combined'] = sorted_combined[:top_k]
        
        return influencers
    
    def _calculate_combined_influence(self, centralities: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate combined influence score across all centrality measures"""
        
        # Weights for different centrality measures in crisis context
        weights = {
            'degree': 0.20,
            'betweenness': 0.25,  # Higher weight for information brokers
            'closeness': 0.15,
            'eigenvector': 0.20,
            'pagerank': 0.20
        }
        
        combined_scores = defaultdict(float)
        total_weight = 0.0
        
        for measure, values in centralities.items():
            weight = weights.get(measure, 0.1)
            total_weight += weight
            
            for node, score in values.items():
                combined_scores[node] += score * weight
        
        # Normalize by total weight
        if total_weight > 0:
            for node in combined_scores:
                combined_scores[node] /= total_weight
        
        return dict(combined_scores)
    
    def calculate_centrality_correlations(self, centralities: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between different centrality measures"""
        logger.info("🔄 Calculating centrality correlations")
        
        if len(centralities) < 2:
            return {}
        
        # Get common nodes
        all_nodes = set()
        for values in centralities.values():
            all_nodes.update(values.keys())
        
        # Create correlation matrix
        correlations = {}
        measures = list(centralities.keys())
        
        for i, measure1 in enumerate(measures):
            correlations[measure1] = {}
            for j, measure2 in enumerate(measures):
                if i <= j:  # Only calculate upper triangle
                    if measure1 == measure2:
                        correlations[measure1][measure2] = 1.0
                    else:
                        # Get values for common nodes
                        values1 = [centralities[measure1].get(node, 0.0) for node in all_nodes]
                        values2 = [centralities[measure2].get(node, 0.0) for node in all_nodes]
                        
                        # Calculate correlation
                        try:
                            correlation = np.corrcoef(values1, values2)[0, 1]
                            if np.isnan(correlation):
                                correlation = 0.0
                        except:
                            correlation = 0.0
                        
                        correlations[measure1][measure2] = correlation
                        
                        # Mirror for lower triangle
                        if measure2 not in correlations:
                            correlations[measure2] = {}
                        correlations[measure2][measure1] = correlation
        
        return correlations
    
    def get_crisis_network_indicators(self, G: nx.Graph, centralities: Dict, communities: Dict, cohesion: Dict) -> Dict[str, float]:
        """Extract crisis-specific network indicators"""
        
        indicators = {
            # Basic network structure
            'network_density': cohesion.get('density', 0.0),
            'clustering_coefficient': cohesion.get('average_clustering', 0.0),
            'connectivity': 1.0 if cohesion.get('is_connected', False) else 0.0,
            
            # Community structure
            'modularity': communities.get('modularity', 0.0),
            'num_communities': communities.get('num_communities', 0),
            'community_fragmentation': communities.get('num_communities', 0) / max(G.number_of_nodes(), 1),
            
            # Centralization indicators
            'degree_centralization': self._calculate_centralization(centralities.get('degree', {})),
            'betweenness_centralization': self._calculate_centralization(centralities.get('betweenness', {})),
            
            # Crisis-specific indicators
            'information_flow_efficiency': 1.0 - cohesion.get('average_shortest_path', 10.0) / 10.0,
            'network_resilience': cohesion.get('lcc_ratio', 1.0),
            'influence_concentration': self._calculate_influence_concentration(centralities)
        }
        
        return indicators
    
    def _calculate_centralization(self, centrality_dict: Dict[str, float]) -> float:
        """Calculate network centralization index"""
        if not centrality_dict:
            return 0.0
        
        values = list(centrality_dict.values())
        if len(values) <= 1:
            return 0.0
        
        max_val = max(values)
        n = len(values)
        
        # Freeman's centralization formula
        numerator = sum(max_val - val for val in values)
        denominator = (n - 1) * (n - 2) if n > 2 else 1
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _calculate_influence_concentration(self, centralities: Dict[str, Dict]) -> float:
        """Calculate how concentrated influence is in the network"""
        if not centralities:
            return 0.0
        
        # Use combined influence if available, otherwise use degree
        if 'combined' in centralities:
            values = list(centralities['combined'].values())
        elif 'degree' in centralities:
            values = list(centralities['degree'].values())
        else:
            return 0.0
        
        if not values:
            return 0.0
        
        # Calculate Gini coefficient as measure of concentration
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n == 0:
            return 0.0
        
        cumsum = np.cumsum(sorted_values)
        total = cumsum[-1]
        
        if total == 0:
            return 0.0
        
        gini = (2 * sum((i + 1) * val for i, val in enumerate(sorted_values))) / (n * total) - (n + 1) / n
        
        return max(0.0, min(1.0, gini))

if __name__ == "__main__":
    # Test SNA metrics
    metrics_calculator = CrisisSNAMetrics()
    
    # Create sample network
    G = nx.erdos_renyi_graph(20, 0.3)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
    
    # Calculate metrics
    centralities = metrics_calculator.calculate_all_centralities(G, 'test')
    communities = metrics_calculator.detect_communities(G, 'test')
    cohesion = metrics_calculator.calculate_network_cohesion(G, 'test')
    influencers = metrics_calculator.identify_influencers(centralities)
    
    print(f"Centralities: {list(centralities.keys())}")
    print(f"Communities: {communities.get('num_communities', 0)}")
    print(f"Density: {cohesion.get('density', 0.0):.3f}")
    
    logger.info("✅ SNA metrics calculator tested successfully!")
