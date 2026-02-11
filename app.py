import json
import os
from flask import Flask, render_template, request, jsonify, Response
from pipeline_manager import PipelineThread, log_queue

app = Flask(__name__)

# Global State
current_thread = None
DEFAULT_CONFIG_PATH = 'default.json'

def load_config():
    """Safely load config, returning empty dict if failed."""
    if os.path.exists(DEFAULT_CONFIG_PATH):
        try:
            with open(DEFAULT_CONFIG_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'POST':
        new_config = request.json
        # Save to disk
        with open(DEFAULT_CONFIG_PATH, 'w') as f:
            json.dump(new_config, f, indent=2)
        return jsonify({"status": "saved"})
    
    return jsonify(load_config())

@app.route('/api/start', methods=['POST'])
def start_pipeline():
    global current_thread
    if current_thread and current_thread.is_alive():
        return jsonify({"status": "error", "message": "Already running"}), 400
    
    config = request.json
    # [FIX] Ensure output directory exists before starting
    if 'output' not in config:
        config['output'] = {'output_dir': './output'}
        
    current_thread = PipelineThread(config)
    current_thread.start()
    return jsonify({"status": "started"})

@app.route('/api/stop', methods=['POST'])
def stop_pipeline():
    global current_thread
    if current_thread and current_thread.is_alive():
        current_thread.stop()
        return jsonify({"status": "stopping"})
    return jsonify({"status": "not_running"})

@app.route('/stream_logs')
def stream_logs():
    def generate():
        while True:
            try:
                # Non-blocking get
                message = log_queue.get(timeout=1) 
                yield f"data: {message}\n\n"
            except:
                # Send heartbeat to keep connection alive
                yield f": heartbeat\n\n" 
                
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)