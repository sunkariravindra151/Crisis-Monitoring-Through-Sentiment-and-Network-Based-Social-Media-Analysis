#!/usr/bin/env python3
"""
Extract 100% authentic data from crisis analysis results
to create completely accurate dashboard
"""

import json
import os

def extract_complete_real_data():
    """Extract all real data from the analysis results"""
    
    # Load the real analysis results
    json_file = "results/outputs/crisis_analysis_complete_20250708_214617.json"
    
    if not os.path.exists(json_file):
        print(f"❌ Error: {json_file} not found!")
        return None
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("🔍 Extracting 100% authentic data...")
    
    # Extract complete real data for each crisis
    real_data = {}
    
    for crisis_type in ['health', 'technological', 'political', 'social']:
        crisis_data = data['detailed_results'][crisis_type]
        
        # Get basic network stats
        basic_stats = crisis_data['network_results']['basic_stats']
        
        # Get all centralities
        centralities = crisis_data['network_results']['centralities']
        
        # Get all real nodes (from degree centrality)
        all_nodes = list(centralities['degree'].keys())
        
        # Get community data
        communities = crisis_data['network_results']['communities']
        
        # Get crisis results
        crisis_results = crisis_data['crisis_results']
        
        # Get data summary
        data_summary = crisis_data['data_summary']
        
        # Sort influencers by degree centrality (descending)
        influencers = sorted(
            [(node, score) for node, score in centralities['degree'].items()],
            key=lambda x: x[1],
            reverse=True
        )[:15]  # Top 15 influencers
        
        real_data[crisis_type] = {
            'severity': crisis_results['severity_score'],
            'nodes': basic_stats['nodes'],
            'edges': basic_stats['edges'],
            'density': basic_stats['density'],
            'users': data_summary['unique_users'],
            'communities': communities['num_communities'],
            'modularity': communities['modularity'],
            'community_sizes': communities['community_sizes'],
            'all_nodes': all_nodes,
            'influencers': [{'name': name, 'score': score} for name, score in influencers],
            'centralities': {
                'degree': centralities['degree'],
                'betweenness': centralities.get('betweenness', {}),
                'closeness': centralities.get('closeness', {}),
                'pagerank': centralities.get('pagerank', {})
            },
            'community_partition': communities['partition'],
            'community_members': communities['communities']
        }
        
        print(f"✅ {crisis_type.title()} Crisis:")
        print(f"   - Nodes: {basic_stats['nodes']}")
        print(f"   - Edges: {basic_stats['edges']}")
        print(f"   - Density: {basic_stats['density']:.3f}")
        print(f"   - Top Influencer: {influencers[0][0]} ({influencers[0][1]:.3f})")
        print(f"   - Communities: {communities['num_communities']}")
    
    return real_data

def generate_authentic_dashboard_data(real_data):
    """Generate JavaScript data for the authentic dashboard"""
    
    js_data = "// 100% AUTHENTIC DATA extracted from crisis analysis results\n"
    js_data += "const authenticRealData = {\n"
    
    for crisis_type, crisis in real_data.items():
        js_data += f"    {crisis_type}: {{\n"
        js_data += f"        severity: {crisis['severity']},\n"
        js_data += f"        nodes: {crisis['nodes']},\n"
        js_data += f"        edges: {crisis['edges']},\n"
        js_data += f"        density: {crisis['density']},\n"
        js_data += f"        users: {crisis['users']},\n"
        js_data += f"        communities: {crisis['communities']},\n"
        js_data += f"        modularity: {crisis['modularity']},\n"
        js_data += f"        communitySizes: {crisis['community_sizes']},\n"
        
        # All real nodes
        js_data += f"        allNodes: [\n"
        for i, node in enumerate(crisis['all_nodes']):
            js_data += f"            \"{node}\""
            if i < len(crisis['all_nodes']) - 1:
                js_data += ","
            js_data += "\n"
        js_data += f"        ],\n"
        
        # Real influencers
        js_data += f"        influencers: [\n"
        for i, inf in enumerate(crisis['influencers']):
            js_data += f"            {{name: \"{inf['name']}\", score: {inf['score']}}}"
            if i < len(crisis['influencers']) - 1:
                js_data += ","
            js_data += "\n"
        js_data += f"        ],\n"
        
        # Real centralities
        js_data += f"        centralities: {{\n"
        js_data += f"            degree: {json.dumps(crisis['centralities']['degree'], indent=12)},\n"
        if crisis['centralities']['betweenness']:
            js_data += f"            betweenness: {json.dumps(crisis['centralities']['betweenness'], indent=12)},\n"
        if crisis['centralities']['pagerank']:
            js_data += f"            pagerank: {json.dumps(crisis['centralities']['pagerank'], indent=12)}\n"
        js_data += f"        }},\n"
        
        # Community partition
        js_data += f"        communityPartition: {json.dumps(crisis['community_partition'], indent=8)}\n"
        
        js_data += f"    }}"
        if crisis_type != 'social':
            js_data += ","
        js_data += "\n"
    
    js_data += "};\n"
    
    return js_data

if __name__ == "__main__":
    print("🚀 Extracting 100% authentic crisis data...")
    
    # Extract real data
    real_data = extract_complete_real_data()
    
    if real_data:
        # Generate JavaScript data
        js_data = generate_authentic_dashboard_data(real_data)
        
        # Save to file
        with open('authentic_data.js', 'w', encoding='utf-8') as f:
            f.write(js_data)
        
        print("\n✅ Authentic data extracted successfully!")
        print("📁 Generated: authentic_data.js")
        print("\n📊 Summary:")
        print(f"   - Total Nodes: {sum(crisis['nodes'] for crisis in real_data.values())}")
        print(f"   - Total Edges: {sum(crisis['edges'] for crisis in real_data.values())}")
        print(f"   - Total Users: {sum(crisis['users'] for crisis in real_data.values())}")
        print(f"   - Total Communities: {sum(crisis['communities'] for crisis in real_data.values())}")
        
    else:
        print("❌ Failed to extract data!")
