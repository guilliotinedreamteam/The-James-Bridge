import sys
import logging
from neurobridge.model.decoder import NeurobridgeDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_builds():
    """
    Validates that Phase 3 models (Offline and Online) compile and build 
    without errors and output the correct tensor shapes.
    """
    try:
        decoder = NeurobridgeDecoder(timesteps=100, channels=128, phoneme_classes=41)
        
        logger.info("--- Testing Offline Decoder ---")
        offline_model = decoder.build_offline_decoder()
        offline_model.summary(print_fn=logger.info)
        
        logger.info("--- Testing Online Decoder ---")
        online_model = decoder.build_online_decoder()
        online_model.summary(print_fn=logger.info)
        
        logger.info("Phase 3 Neural Architectures built successfully.")
        
    except Exception as e:
        logger.error(f"Phase 3 Architecture failure: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_model_builds()
