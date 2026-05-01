"""
NeuroBridge Flask API

Provides the create_app factory used by tests and the CLI
to serve the BCI clinical endpoints.
"""

import datetime
import logging

import numpy as np
from flask import Flask, jsonify, request

from neurobridge.config import Config

logger = logging.getLogger(__name__)

_model = None  # placeholder for a loaded model


def create_app():
    """Application factory – returns a configured Flask app."""
    app = Flask(__name__)

    # ── Health ──────────────────────────────────────────────
    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "ok",
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "version": Config.__dict__.get("API_VERSION", "1.0.0"),
                "model_loaded": _model is not None,
            }
        )

    # ── Process ─────────────────────────────────────────────
    @app.route("/api/process", methods=["POST"])
    def process():
        ct = request.content_type or ""
        if "application/json" not in ct:
            return (
                jsonify(
                    {
                        "error": "Content-Type must be application/json",
                        "code": "BAD_CONTENT_TYPE",
                    }
                ),
                400,
            )

        raw = request.get_data(as_text=True)
        if not raw or not raw.strip():
            return jsonify({"error": "Empty body", "code": "EMPTY_BODY"}), 400

        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Malformed JSON", "code": "MALFORMED_JSON"}), 400

        signal = data.get("signal")

        # Validate signal presence and type
        if signal is None:
            return (
                jsonify({"error": "Missing 'signal' field", "code": "MISSING_SIGNAL"}),
                400,
            )

        if not isinstance(signal, list):
            return (
                jsonify(
                    {"error": "'signal' must be a list", "code": "INVALID_SIGNAL_TYPE"}
                ),
                400,
            )

        if len(signal) == 0:
            return (
                jsonify(
                    {"error": "'signal' must not be empty", "code": "EMPTY_SIGNAL"}
                ),
                400,
            )

        # Length validation
        expected = Config.NUM_FEATURES
        if len(signal) < expected:
            return (
                jsonify(
                    {
                        "error": f"Signal length {len(signal)} < {expected}",
                        "code": "SIGNAL_LENGTH_MISMATCH",
                    }
                ),
                400,
            )

        if len(signal) > expected:
            return (
                jsonify(
                    {
                        "error": f"Signal length {len(signal)} > {expected}",
                        "code": "SIGNAL_LENGTH_MISMATCH",
                    }
                ),
                400,
            )

        # NaN check
        try:
            arr = np.array(signal, dtype=np.float64)
            if np.any(np.isnan(arr)):
                pass  # allow through to model / 503
        except (ValueError, TypeError):
            return (
                jsonify(
                    {
                        "error": "Non-numeric values in signal",
                        "code": "INVALID_SIGNAL_VALUES",
                    }
                ),
                400,
            )

        if _model is None:
            return (
                jsonify({"error": "Model is not loaded", "code": "MODEL_NOT_LOADED"}),
                503,
            )

        # Inference (would go here if model loaded)
        return jsonify({"result": "ok"})

    # ── Analyze ─────────────────────────────────────────────
    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        payload = request.get_json(silent=True)
        if payload is None or "data" not in payload or payload["data"] is None:
            return (
                jsonify({"error": "Missing 'data' field", "code": "MISSING_DATA"}),
                400,
            )

        raw_data = payload["data"]
        analysis_type = payload.get("analysis_type", "time_domain")

        try:
            arr = np.array(raw_data, dtype=np.float64)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid data", "code": "INVALID_DATA"}), 400

        # Ensure at least 1-D
        if arr.ndim == 0:
            return (
                jsonify({"error": "Scalar data not supported", "code": "INVALID_DATA"}),
                400,
            )

        # Compute shape (list-of-lists → 2-D, flat list → 1-D)
        shape = list(arr.shape)

        stats = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

        results = {
            "analysis_type": analysis_type,
            "shape": shape,
            "statistics": stats,
        }

        if analysis_type == "frequency_domain":
            power = float(np.sum(np.abs(arr) ** 2))
            results["frequency_analysis"] = {
                "total_power": power,
            }

        return jsonify(
            {
                "results": results,
                "visualization_data": {"available": True},
            }
        )

    # ── Error handlers ──────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "code": "NOT_FOUND"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return (
            jsonify({"error": "Method not allowed", "code": "METHOD_NOT_ALLOWED"}),
            405,
        )

    return app
