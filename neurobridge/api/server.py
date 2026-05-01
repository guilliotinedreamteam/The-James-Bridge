import collections
import logging
import time

import numpy as np

try:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
except ImportError:
    Starlette = None  # type: ignore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_online_model = None
_actuator = None


# Global rolling buffer for CNN-LSTM temporal context
# We buffer 10 timesteps by default for the temporal convolutions
_temporal_buffer = None
BUFFER_SIZE = 10


async def predict_frame(request):
    """
    Ingests a single frame of data, runs inference, and returns phoneme probabilities.
    Phase 9 Latency Optimization: Bypassing `predict` overhead in favor of direct tensor invocation.
    Now supports rolling temporal buffer for SOTA CNN-LSTM hybrid.
    """
    global _temporal_buffer

    try:
        start_time = time.perf_counter()

        payload = await request.json()
        data_list = payload.get("data")

        if not data_list:
            return JSONResponse(
                {"error": "Missing 'data' array in payload."}, status_code=400
            )

        arr = np.array(data_list)

        # Dynamic channel support
        expected_channels = _online_model.input_shape[-1]
        if arr.shape != (1, expected_channels):
            return JSONResponse(
                {"error": f"Expected shape (1, {expected_channels}), got {arr.shape}"},
                status_code=400,
            )

        # Initialize buffer with zeros if it doesn't exist
        if _temporal_buffer is None:
            _temporal_buffer = collections.deque(
                [np.zeros((expected_channels,)) for _ in range(BUFFER_SIZE)],
                maxlen=BUFFER_SIZE,
            )

        # Append the new frame (squeeze out the batch dim for the deque)
        _temporal_buffer.append(np.squeeze(arr))

        # Stack the buffer into a tensor of shape (batch=1, timesteps=10, channels=128)
        buffer_array = np.array(_temporal_buffer)
        tensor_input = np.expand_dims(buffer_array, axis=0)

        # PHASE 9: Latency Optimization.
        # model.predict() has massive overhead. Direct invocation is vastly faster for real-time.
        probs_tensor = _online_model(tensor_input, training=False)

        probs_array = np.squeeze(probs_tensor.numpy())
        top_phoneme_id = int(np.argmax(probs_array))
        confidence = float(np.max(probs_array))
        probs = probs_array.tolist()

        # ACTUATION TRIGGER: Phase 7
        actuated = False
        if _actuator and confidence > 0.8:
            actuated = _actuator.send_command(top_phoneme_id)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return JSONResponse(
            {
                "phoneme_id": top_phoneme_id,
                "confidence": confidence,
                "actuated": actuated,
                "latency_ms": round(latency_ms, 2),
                "probabilities": probs,
            }
        )

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def health_check(request):
    return JSONResponse(
        {
            "status": "Online",
            "model": "Unidirectional LSTM",
            "actuation_mode": _actuator.mode if _actuator else "disabled",
        }
    )


def build_api(model, actuator=None):
    """Factory to build the Starlette application with model and actuator injection."""
    if Starlette is None:
        raise ImportError(
            "Starlette is not installed. Run 'pip install starlette uvicorn'."
        )

    global _online_model, _actuator
    _online_model = model
    _actuator = actuator

    routes = [
        Route("/predict", endpoint=predict_frame, methods=["POST"]),
        Route("/health", endpoint=health_check, methods=["GET"]),
    ]

    app = Starlette(debug=False, routes=routes)
    return app
