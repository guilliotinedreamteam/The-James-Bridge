import logging
import sys
import numpy as np

try:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
except ImportError:
    Starlette = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_online_model = None
_actuator = None

async def predict_frame(request):
    """
    Ingests a single frame of data, runs inference, and returns phoneme probabilities.
    Now triggers Phase 7 Actuation on high-confidence detections.
    """
    try:
        payload = await request.json()
        data_list = payload.get("data")
        
        if not data_list:
            return JSONResponse({"error": "Missing 'data' array in payload."}, status_code=400)
            
        arr = np.array(data_list)
        
        # Dynamic channel support: the model's input layer already knows its shape.
        # We just need to ensure the incoming data matches the model's expected channels.
        expected_channels = _online_model.input_shape[-1]
        if arr.shape != (1, expected_channels):
            return JSONResponse({"error": f"Expected shape (1, {expected_channels}), got {arr.shape}"}, status_code=400)
            
        tensor_input = np.expand_dims(arr, axis=0)
        probs = _online_model.predict(tensor_input, verbose=0)
        probs = np.squeeze(probs).tolist()
        top_phoneme_id = int(np.argmax(probs))
        confidence = float(np.max(probs))
        
        # ACTUATION TRIGGER: Phase 7
        # If prediction confidence > 0.8, we actuate the prosthetic.
        actuated = False
        if _actuator and confidence > 0.8:
            actuated = _actuator.send_command(top_phoneme_id)
        
        return JSONResponse({
            "phoneme_id": top_phoneme_id,
            "confidence": confidence,
            "actuated": actuated,
            "probabilities": probs
        })
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def health_check(request):
    return JSONResponse({
        "status": "Online", 
        "model": "Unidirectional LSTM",
        "actuation_mode": _actuator.mode if _actuator else "disabled"
    })

def build_api(model, actuator=None):
    """Factory to build the Starlette application with model and actuator injection."""
    if Starlette is None:
        raise ImportError("Starlette is not installed. Run 'pip install starlette uvicorn'.")
        
    global _online_model, _actuator
    _online_model = model
    _actuator = actuator

    routes = [
        Route("/predict", endpoint=predict_frame, methods=["POST"]),
        Route("/health", endpoint=health_check, methods=["GET"])
    ]
    
    app = Starlette(debug=False, routes=routes)
    return app
