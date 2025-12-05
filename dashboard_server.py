import redis
import json
import pickle
import time
from flask import Flask, jsonify, request, render_template, Response
from flask_cors import CORS
from staff_filter import StaffFilter
import numpy as np

app = Flask(__name__, template_folder='templates')
CORS(app)

r_bytes = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
r_json = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    return Response(generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames(cam_id):
    while True:
        frame_data = r_bytes.get(f"live_feed:{cam_id}")
        if frame_data:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.04)

@app.route('/api/daily_stats')
def get_daily_stats():
    query_date = request.args.get('date', time.strftime("%Y-%m-%d"))
    pattern = f"mct:stats:*:{query_date}"
    keys = list(r_json.scan_iter(pattern))
    response_data = {"date": query_date, "cameras": {}}

    for key in keys:
        cam_name = key.split(':')[2]
        stats = r_json.hgetall(key)
        unique = r_json.pfcount(f"mct:unique:{cam_name}:{query_date}")
        hourly = {k.split('_')[0]: int(v) for k, v in stats.items() if k.startswith('H')}
        
        response_data["cameras"][cam_name] = {
            "total_tripwire": int(stats.get('total_tripwire', 0)),
            "unique_visitors": unique,
            "hourly_tripwire": hourly
        }
    return jsonify(response_data)

@app.route('/api/staff/<group_id>', methods=['GET'])
def get_staff(group_id):
    sf = StaffFilter(r_bytes, group_id)
    return jsonify(sf.list_all_staff())

@app.route('/api/staff/delete', methods=['POST'])
def delete_staff():
    data = request.json
    sf = StaffFilter(r_bytes, data['group_id'])
    return jsonify({"success": sf.remove_staff_member(data['staff_id'])})

# --- NEW RESET ROUTE ---
@app.route('/api/staff/reset', methods=['POST'])
def reset_staff_db():
    data = request.json
    sf = StaffFilter(r_bytes, data['group_id'])
    success = sf.delete_all_staff()
    return jsonify({"success": success})

@app.route('/api/staff/calibrate', methods=['POST'])
def calibrate_staff():
    try:
        data = request.json
        gid, group = data['track_id'], data['group_id']
        
        # Try the new gallery_core format first (Core Set approach)
        core_data = r_bytes.get(f"gallery_core:{gid}")
        
        if core_data:
            # New format: Core set is a list of vectors
            core_set = pickle.loads(core_data)
            if core_set and len(core_set) > 0:
                # Use the mean of the core set as the representative vector
                vec = np.mean(core_set, axis=0)
                vec = vec / (np.linalg.norm(vec) + 1e-6)  # Re-normalize
            else:
                return jsonify({"error": "ID has empty gallery"}), 404
        else:
            # Fallback: Try legacy format for backwards compatibility
            legacy_item = r_bytes.lindex(f"gallery:{gid}", -1)
            if not legacy_item:
                return jsonify({"error": "ID inactive or not found"}), 404
            vec = pickle.loads(legacy_item)['v']
        
        sf = StaffFilter(r_bytes, group)
        sf.add_staff_member(str(gid), data['name'], data['role'], vec)
        return jsonify({"success": True})
    except Exception as e: 
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)