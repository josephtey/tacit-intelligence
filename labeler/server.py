"""Backend server for E2E runs viewer and live inference.

Run:  python labeler/server.py
Serves on port 8765.
"""

from flask import Flask, jsonify, request, send_from_directory, abort, Response
import json
import os
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "pipette_well_dataset")
E2E_RUNS_DIR = os.path.join(BASE_DIR, "experiments", "e2e_runs")
INFERENCE_DIR = os.path.join(BASE_DIR, "experiments", "inference_sessions")


# ── End-to-End runs endpoints ──

@app.route("/api/e2e_runs")
def list_e2e_runs():
    if not os.path.isdir(E2E_RUNS_DIR):
        return jsonify([])
    entries = []
    for entry in os.listdir(E2E_RUNS_DIR):
        run_folder = os.path.join(E2E_RUNS_DIR, entry)
        if not os.path.isdir(run_folder):
            continue
        results_path = os.path.join(run_folder, "results.json")
        if not os.path.exists(results_path):
            continue
        mtime = os.path.getmtime(results_path)
        entries.append((entry, results_path, mtime))
    # Sort by modification time, newest first
    entries.sort(key=lambda x: x[2], reverse=True)
    runs = []
    for entry, results_path, _ in entries:
        with open(results_path) as f:
            data = json.load(f)
        runs.append({
            "name": entry,
            "summary": data.get("summary", {}),
        })
    return jsonify(runs)


@app.route("/api/e2e_runs/<run_name>")
def get_e2e_run(run_name):
    run_folder = os.path.join(E2E_RUNS_DIR, run_name)
    results_path = os.path.join(run_folder, "results.json")
    if not os.path.exists(results_path):
        abort(404)
    with open(results_path) as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/e2e_images/<run_name>/<filename>")
def serve_e2e_image(run_name, filename):
    run_folder = os.path.join(E2E_RUNS_DIR, run_name)
    if not os.path.isdir(run_folder):
        abort(404)
    return send_from_directory(run_folder, filename)


# ── Live inference endpoints ──

@app.route("/api/inference", methods=["POST"])
def run_inference():
    """Accept video uploads, run E2E pipeline, return results."""
    import uuid
    import sys
    sys.path.insert(0, BASE_DIR)

    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(INFERENCE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Collect uploaded file pairs
    pairs = []

    # Check for single pair upload
    fpv_file = request.files.get("fpv_video")
    tv_file = request.files.get("topview_video")
    if fpv_file and tv_file:
        fpv_path = os.path.join(session_dir, fpv_file.filename)
        tv_path = os.path.join(session_dir, tv_file.filename)
        fpv_file.save(fpv_path)
        tv_file.save(tv_path)
        pairs.append((fpv_path, tv_path))

    # Check for batch upload (numbered pairs)
    i = 0
    while True:
        fpv = request.files.get(f"fpv_video_{i}")
        tv = request.files.get(f"topview_video_{i}")
        if not fpv or not tv:
            break
        fpv_path = os.path.join(session_dir, fpv.filename)
        tv_path = os.path.join(session_dir, tv.filename)
        fpv.save(fpv_path)
        tv.save(tv_path)
        pairs.append((fpv_path, tv_path))
        i += 1

    if not pairs:
        return jsonify({"error": "No video files uploaded"}), 400

    # Run pipeline
    try:
        from src.e2e_pipeline import EndToEndPipeline
        pipeline = EndToEndPipeline(generate_overlays=True)
    except Exception as e:
        return jsonify({"error": f"Failed to load pipeline: {e}"}), 500

    results = []
    for fpv_path, tv_path in pairs:
        try:
            result = pipeline.predict(
                fpv_path, tv_path,
                overlay_dir=session_dir,
            )
        except Exception as e:
            result = {
                "clip_id_FPV": os.path.splitext(os.path.basename(fpv_path))[0],
                "clip_id_Topview": os.path.splitext(os.path.basename(tv_path))[0],
                "wells_prediction": [],
                "stages": {},
                "pipette_type": None,
                "status": "error",
                "error": str(e),
            }
        # Add overlay info (only for files that were actually generated)
        clip_name = os.path.splitext(os.path.basename(fpv_path))[0].replace("_FPV", "")
        overlay_candidates = {
            "commit": f"{clip_name}_s1_commit.jpg",
            "corners": f"{clip_name}_s2_corners.jpg",
            "warped": f"{clip_name}_s3_warped.jpg",
            "summary": f"{clip_name}_summary.jpg",
        }
        result["overlays"] = {
            key: fname
            for key, fname in overlay_candidates.items()
            if os.path.exists(os.path.join(session_dir, fname))
        }
        results.append(result)

    # Make results JSON-serializable (strip numpy types)
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = [sanitize(r) for r in results]

    # Save results
    output = {"session_id": session_id, "results": results}
    with open(os.path.join(session_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=2)

    return jsonify(output)


@app.route("/api/inference/<session_id>")
def get_inference_result(session_id):
    session_dir = os.path.join(INFERENCE_DIR, session_id)
    results_path = os.path.join(session_dir, "results.json")
    if not os.path.exists(results_path):
        abort(404)
    with open(results_path) as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/inference_images/<session_id>/<filename>")
def serve_inference_image(session_id, filename):
    session_dir = os.path.join(INFERENCE_DIR, session_id)
    if not os.path.isdir(session_dir):
        abort(404)
    return send_from_directory(session_dir, filename)


if __name__ == "__main__":
    print(f"E2E runs:     {E2E_RUNS_DIR}")
    print(f"Inference:    {INFERENCE_DIR}")
    app.run(host="0.0.0.0", port=8765, debug=True)
