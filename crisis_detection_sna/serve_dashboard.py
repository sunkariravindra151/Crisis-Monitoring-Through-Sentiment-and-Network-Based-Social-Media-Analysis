#!/usr/bin/env python3
"""
Simple HTTP server to serve the SNA dashboard files
This resolves CORS issues when loading local files
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def serve_dashboard(port=8000):
    """
    Start a simple HTTP server to serve the dashboard files
    """
    # Change to the current directory
    os.chdir(Path(__file__).parent)
    
    # Create handler
    handler = http.server.SimpleHTTPRequestHandler
    
    # Add CORS headers
    class CORSRequestHandler(handler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            super().end_headers()
    
    try:
        with socketserver.TCPServer(("", port), CORSRequestHandler) as httpd:
            print(f"🚀 Starting SNA Dashboard Server...")
            print(f"📊 Server running at: http://localhost:{port}")
            print(f"🌐 Dashboard URL: http://localhost:{port}/working_sna_dashboard.html")
            print(f"📈 Enhanced Dashboard: http://localhost:{port}/enhanced_sna_dashboard.html")
            print(f"📋 Features Documentation: http://localhost:{port}/SNA_Dashboard_Features.md")
            print(f"\n🔧 Press Ctrl+C to stop the server")
            
            # Open the dashboard in browser
            dashboard_url = f"http://localhost:{port}/working_sna_dashboard.html"
            print(f"\n🌟 Opening dashboard in browser: {dashboard_url}")
            webbrowser.open(dashboard_url)
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
        sys.exit(0)
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use. Trying port {port + 1}...")
            serve_dashboard(port + 1)
        else:
            print(f"❌ Error starting server: {e}")
            sys.exit(1)

if __name__ == "__main__":
    # Check if port is provided as argument
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid port number. Using default port 8000.")
    
    # Check if dashboard files exist
    dashboard_files = [
        "working_sna_dashboard.html",
        "enhanced_sna_dashboard.html", 
        "sna_comprehensive_dashboard.html"
    ]
    
    existing_files = [f for f in dashboard_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No dashboard files found in current directory!")
        print("📁 Make sure you're running this script from the crisis_detection_sna directory")
        sys.exit(1)
    
    print(f"✅ Found dashboard files: {', '.join(existing_files)}")
    
    # Start the server
    serve_dashboard(port)
