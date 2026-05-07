import asyncio
import os
import json
import time
from datetime import datetime
from pathlib import Path
from llmeter.endpoints import OpenAICompletionEndpoint
from llmeter.experiments import LoadTest
from llmeter.callbacks import CostModel
from llmeter.callbacks.cost import dimensions
from dotenv import load_dotenv

load_dotenv()

endpoint = OpenAICompletionEndpoint(
    model_id="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

payload = endpoint.create_payload(
    "Explain why latency percentiles matter in production AI systems.",
    max_tokens=200,
)

# Cost model
cost_model = CostModel(
    request_dims=[
        dimensions.InputTokens(price_per_million=2.50),
        dimensions.OutputTokens(price_per_million=10.00),
    ]
)

# Shared state for real-time updates
class RealtimeStats:
    def __init__(self):
        self.start_time = None
        self.completed_requests = 0
        self.total_requests = 0
        self.current_latency = 0.0
        self.current_cost = 0.0
        self.errors = 0
        self.status = "idle"
        self.clients = 0
        self.stop_requested = False
    
    def to_dict(self):
        elapsed = (time.time() - self.start_time) if self.start_time else 0
        return {
            "status": self.status,
            "completed_requests": self.completed_requests,
            "total_requests": self.total_requests,
            "progress": f"{(self.completed_requests/self.total_requests*100):.1f}%" if self.total_requests > 0 else "0%",
            "current_latency": f"{self.current_latency:.2f}s",
            "current_cost": f"${self.current_cost:.6f}",
            "errors": self.errors,
            "clients": self.clients,
            "stop_requested": self.stop_requested,
            "elapsed_time": f"{elapsed:.1f}s",
            "requests_per_second": f"{self.completed_requests/elapsed:.1f}" if elapsed > 0 else "0"
        }

stats = RealtimeStats()

# Custom callback for real-time updates
class RealtimeCallback:
    def __init__(self):
        stats.total_requests = 5 * 2  # 5 clients * 2 requests per client
        stats.clients = 5
        self._save_stats()
    
    async def before_run(self, run_config):
        self._save_stats()
    
    async def before_invoke(self, payload):
        pass
    
    async def after_invoke(self, response):
        """Called after each request completes"""
        stats.completed_requests += 1
        stats.current_latency = response.time_to_last_token or 0
        stats.current_cost += (response.num_tokens_input * 2.50 / 1_000_000) + (response.num_tokens_output * 10.00 / 1_000_000)
        if response.error:
            stats.errors += 1
        # Update stats file
        self._save_stats()
    
    async def after_run(self, result):
        self._save_stats()
    
    def _save_stats(self):
        stats_file = Path("./results/realtime_stats.json")
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with stats_file.open("w") as f:
            json.dump(stats.to_dict(), f)

async def main():
    print("Starting Real-Time Monitoring Dashboard...")
    print(f"Open http://localhost:8000/dashboard.html in your browser")
    
    # Start simple HTTP server for dashboard
    import http.server
    import socketserver
    import threading
    
    PORT = 8000
    stop_event = threading.Event()
    
    class DashboardHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/dashboard.html":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Load Test Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2rem; margin-bottom: 10px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
        .card h3 { color: #f1f5f9; margin-bottom: 10px; font-size: 1rem; }
        .metric { font-size: 2rem; font-weight: 600; color: #3b82f6; }
        .label { color: #94a3b8; font-size: 0.9rem; }
        .status { padding: 10px 20px; border-radius: 8px; font-weight: 600; text-align: center; margin-bottom: 20px; }
        .status.running { background: #22c55e; color: white; }
        .status.idle { background: #64748b; color: white; }
        .status.completed { background: #3b82f6; color: white; }
        .status.stopped { background: #ef4444; color: white; }
        .actions { display: flex; justify-content: center; margin-bottom: 20px; }
        .stop-button { background: #dc2626; border: none; border-radius: 8px; color: white; cursor: pointer; font-size: 1rem; font-weight: 600; padding: 12px 24px; }
        .stop-button:hover { background: #b91c1c; }
        .stop-button:disabled { background: #64748b; cursor: not-allowed; }
        .chart-container { position: relative; height: 300px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Real-Time Load Test Dashboard</h1>
            <div id="status" class="status idle">Idle</div>
            <div class="actions">
                <button id="stopButton" class="stop-button" onclick="stopTest()">Stop Test</button>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>Progress</h3>
                <div class="metric" id="progress">0%</div>
                <div class="label">Completed</div>
            </div>
            <div class="card">
                <h3>Requests</h3>
                <div class="metric" id="completed">0</div>
                <div class="label">Completed / <span id="total">0</span></div>
            </div>
            <div class="card">
                <h3>Current Latency</h3>
                <div class="metric" id="latency">0s</div>
                <div class="label">Latest Request</div>
            </div>
            <div class="card">
                <h3>Total Cost</h3>
                <div class="metric" id="cost">$0.000000</div>
                <div class="label">Accumulated</div>
            </div>
            <div class="card">
                <h3>Errors</h3>
                <div class="metric" id="errors">0</div>
                <div class="label">Failed Requests</div>
            </div>
            <div class="card">
                <h3>Throughput</h3>
                <div class="metric" id="throughput">0 req/s</div>
                <div class="label">Requests Per Second</div>
            </div>
        </div>
        
        <div class="card">
            <h3>Latency Over Time</h3>
            <div class="chart-container">
                <canvas id="latencyChart"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        const latencyData = [];
        const ctx = document.getElementById('latencyChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Latency (s)',
                    data: latencyData,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
                },
                plugins: { legend: { display: false } }
            }
        });
        
        async function updateDashboard() {
            try {
                const response = await fetch('/realtime_stats.json');
                const data = await response.json();
                
                document.getElementById('status').textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
                document.getElementById('status').className = `status ${data.status}`;
                document.getElementById('progress').textContent = data.progress;
                document.getElementById('completed').textContent = data.completed_requests;
                document.getElementById('total').textContent = data.total_requests;
                document.getElementById('latency').textContent = data.current_latency;
                document.getElementById('cost').textContent = data.current_cost;
                document.getElementById('errors').textContent = data.errors;
                document.getElementById('throughput').textContent = data.requests_per_second;
                document.getElementById('stopButton').disabled = data.status === 'completed' || data.status === 'stopped';
                
                if (data.current_latency !== '0.00s') {
                    const latency = parseFloat(data.current_latency);
                    latencyData.push(latency);
                    chart.data.labels.push(latencyData.length);
                    chart.update('none');
                }
            } catch (e) {
                console.log('Waiting for test to start...');
            }
        }
        
        async function stopTest() {
            const button = document.getElementById('stopButton');
            button.disabled = true;
            button.textContent = 'Stopping...';
            await fetch('/stop', { method: 'POST' });
        }
        
        setInterval(updateDashboard, 500);
    </script>
</body>
</html>
                """
                self.wfile.write(html.encode())
            elif self.path == "/realtime_stats.json":
                stats_file = Path("./results/realtime_stats.json")
                if stats_file.exists():
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    with stats_file.open("rb") as f:
                        self.wfile.write(f.read())
                else:
                    self.send_response(404)
                    self.end_headers()
            else:
                super().do_GET()
        
        def do_POST(self):
            if self.path == "/stop":
                stats.stop_requested = True
                stats.status = "stopped"
                stop_event.set()
                RealtimeCallback()._save_stats()
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"stopped"}')
            else:
                self.send_response(404)
                self.end_headers()
    
    # Start server in background thread
    def start_server():
        with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
            print(f"\nDashboard server running at http://localhost:{PORT}/dashboard.html")
            httpd.serve_forever()
    
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait a bit for server to start
    await asyncio.sleep(2)
    
    # Run load test with real-time updates
    stats.status = "running"
    stats.start_time = time.time()
    
    realtime_callback = RealtimeCallback()
    
    load_test = LoadTest(
        endpoint=endpoint,
        payload=payload,
        sequence_of_clients=[5],
        output_path="./results/realtime",
        callbacks=[cost_model, realtime_callback]
    )
    
    load_test_task = asyncio.create_task(load_test.run())
    
    while not load_test_task.done():
        if stop_event.is_set():
            stats.status = "stopped"
            stats.stop_requested = True
            realtime_callback._save_stats()
            load_test_task.cancel()
            try:
                await load_test_task
            except asyncio.CancelledError:
                pass
            break
        await asyncio.sleep(0.2)
    
    if not load_test_task.cancelled() and not stop_event.is_set():
        await load_test_task
        stats.status = "completed"
        realtime_callback._save_stats()
    
    print("\nLoad test completed!")
    print(f"Dashboard still available at http://localhost:{PORT}/dashboard.html")
    print("Press Ctrl+C to stop the server")
    
    # Keep server running
    try:
        while not stop_event.is_set():
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    asyncio.run(main())
