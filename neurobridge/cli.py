import argparse
import sys
import logging
import numpy as np
from neurobridge.data.ingestion import ECoGIngestionPipeline
from neurobridge.processing.signal import SignalProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lazy import Phase 3 & 4 architecture to avoid crashing if TensorFlow isn't present
def get_decoder_and_trainer():
    try:
        from neurobridge.model.decoder import NeurobridgeDecoder
        from neurobridge.training.trainer import ModelTrainer
        return NeurobridgeDecoder, ModelTrainer
    except ImportError:
        logger.error("Phase 3/4 dependencies missing. Ensure TensorFlow is installed via 'pip install tensorflow'.")
        sys.exit(1)

def handle_info(args):
    """
    Executes the 'info' command to output system capabilities and boundaries.
    """
    logger.info("Neurobridge V1 System Info")
    logger.info("Operating Mode: Medical-Grade Offline Training / Real-Time Inference")
    logger.info("Supported Ingestion Formats: .edf, .bdf, .vhdr")
    logger.info("Target Input Channels: Dynamic (ECoG/EEG)")
    logger.info("Target Decoder Output Classes: 41 (Phonemes)")
    logger.info("Status: Phase 1-5 Active. Hyperparameter Tuning Suite Integrated.")

def _process_file(file_path: str) -> np.ndarray:
    """Helper to process an clinical data file and return shaped sequences."""
    pipeline = ECoGIngestionPipeline(expected_channels=1) # Allow dynamic
    raw_data = pipeline.load_medical_dataset(file_path)
    data_array = pipeline.extract_numpy_arrays(raw_data)
    processor = SignalProcessor()
    downsampled = processor.downsample_signals(data_array, current_freq=int(raw_data.info['sfreq']))
    normalized = processor.z_score_normalize(downsampled)
    shaped = processor.shape_sequences(normalized)
    return shaped

def handle_ingest(args):
    """
    Executes the 'ingest' command to test medical data loading.
    """
    try:
        shaped = _process_file(args.file)
        logger.info(f"Ingestion & Processing successful. Final sequence shape: {shaped.shape}")
        
        # If decode flag is set, run Phase 3 dry-run
        if getattr(args, 'decode', False):
            NeurobridgeDecoder, _ = get_decoder_and_trainer()
            batch_size, timesteps, channels = shaped.shape
            decoder = NeurobridgeDecoder(timesteps=timesteps, channels=channels)
            logger.info("Instantiating Offline Model for dry-run prediction...")
            model = decoder.build_offline_decoder()
            predictions = model.predict(shaped)
            logger.info(f"Phase 3 Decoding successful. Output prediction tensor shape: {predictions.shape}")
            
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        sys.exit(1)

def handle_train(args):
    """
    Executes the offline training loop on the provided medical data.
    """
    # Step 1: Ingest and shape the continuous neural data
    pipeline = ECoGIngestionPipeline(expected_channels=1)
    raw_data = pipeline.load_medical_dataset(args.file)
    data_array = pipeline.extract_numpy_arrays(raw_data)
    
    processor = SignalProcessor()
    downsampled = processor.downsample_signals(data_array, current_freq=int(raw_data.info['sfreq']))
    normalized = processor.z_score_normalize(downsampled)
    shaped = processor.shape_sequences(normalized)
    
    batch_size, timesteps, channels = shaped.shape
    NeurobridgeDecoder, ModelTrainer = get_decoder_and_trainer()
    
    # Step 2: Target Label Alignment
    if getattr(args, 'labels', None):
        from neurobridge.data.alignment import LabelAligner
        aligner = LabelAligner(
            original_sfreq=int(raw_data.info['sfreq']),
            target_sfreq=processor.target_freq,
            target_timesteps=timesteps,
            phoneme_classes=41
        )
        df = aligner.load_events(args.labels)
        y_train = aligner.align_to_tensor(df, total_original_samples=data_array.shape[1])
        logger.info("Successfully aligned clinical labels to ground-truth tensor.")
    else:
        logger.warning("No clinical phoneme labels provided (--labels). Generating synthetic targets for training validation.")
        random_phonemes = np.random.randint(0, 41, size=(batch_size, timesteps))
        y_train = np.eye(41)[random_phonemes]
        
    # Step 3: Neural Decoding Architecture
    decoder = NeurobridgeDecoder(timesteps=timesteps, channels=channels)
    model = decoder.build_offline_decoder()
    
    # Step 4: Training Execution
    trainer = ModelTrainer(model=model, checkpoint_dir="checkpoints")
    trainer.train(x_train=shaped, y_train=y_train, epochs=args.epochs, batch_size=8, model_name="v1_offline_decoder")
    
    logger.info("Phase 4 Training pipeline execution complete.")

def handle_tune(args):
    """
    Executes a hyperparameter tuning grid search on the clinical data.
    """
    from neurobridge.training.tuner import HyperTuner
    from neurobridge.data.alignment import LabelAligner
    
    pipeline = ECoGIngestionPipeline(expected_channels=1)
    raw_data = pipeline.load_medical_dataset(args.file)
    data_array = pipeline.extract_numpy_arrays(raw_data)
    
    processor = SignalProcessor()
    downsampled = processor.downsample_signals(data_array, current_freq=int(raw_data.info['sfreq']))
    normalized = processor.z_score_normalize(downsampled)
    shaped = processor.shape_sequences(normalized)
    
    if not args.labels:
        raise ValueError("Legitimate clinical labels (--labels) are required for tuning.")
        
    aligner = LabelAligner(
        original_sfreq=int(raw_data.info['sfreq']),
        target_sfreq=processor.target_freq,
        target_timesteps=shaped.shape[1],
        phoneme_classes=41
    )
    df = aligner.load_events(args.labels)
    y_data = aligner.align_to_tensor(df, total_original_samples=data_array.shape[1])
    
    tuner = HyperTuner(x_data=shaped, y_data=y_data)
    
    # Execute grid search over standard tuning ranges
    tuner.execute_grid_search(
        lstm_options=[128, 256], 
        dropout_options=[0.2, 0.4], 
        epochs=args.epochs
    )

def handle_serve(args):
    """
    Executes the real-time inference server (Phase 5).
    Now integrated with Phase 7 Actuation.
    """
    logger.info("Initializing Phase 5 Real-Time Inference Server...")
    try:
        import uvicorn
        from neurobridge.api.server import build_api
        from neurobridge.actuation.interface import ProstheticInterface
    except ImportError as e:
        logger.error(f"Required dependencies not installed. Error: {e}")
        sys.exit(1)
        
    try:
        NeurobridgeDecoder, _ = get_decoder_and_trainer()
        
        # Instantiate the online Unidirectional LSTM
        decoder = NeurobridgeDecoder()
        online_model = decoder.build_online_decoder()
        
        # ACTUATION INTERFACE: Phase 7
        actuator = ProstheticInterface(mode=args.actuation)
        logger.info(f"Phase 7 Actuation enabled in {args.actuation} mode.")
        
        # Build the Starlette app and inject the model + actuator
        app = build_api(online_model, actuator=actuator)
        
        logger.info(f"Server starting on {args.host}:{args.port}. Waiting for ECoG streams...")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Neurobridge V1 - Medical BCI Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: info
    parser_info = subparsers.add_parser("info", help="Display system capabilities and configuration")
    parser_info.set_defaults(func=handle_info)

    # Command: ingest
    parser_ingest = subparsers.add_parser("ingest", help="Ingest a clinical ECoG dataset (.edf or .bdf)")
    parser_ingest.add_argument("--file", type=str, required=True, help="Path to the .edf or .bdf medical dataset")
    parser_ingest.add_argument("--decode", action="store_true", help="Run Phase 3 Neural Decoding (dry-run) after processing")
    parser_ingest.set_defaults(func=handle_ingest)
    
    # Command: train
    parser_train = subparsers.add_parser("train", help="Train the offline Bidirectional LSTM model")
    parser_train.add_argument("--file", type=str, required=True, help="Path to the .edf or .bdf medical training dataset")
    parser_train.add_argument("--labels", type=str, default=None, help="Path to the BIDS .tsv events file for clinical target alignment")
    parser_train.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser_train.set_defaults(func=handle_train)

    # Command: tune
    parser_tune = subparsers.add_parser("tune", help="Run hyperparameter tuning on clinical data")
    parser_tune.add_argument("--file", type=str, required=True, help="Path to the dataset file")
    parser_tune.add_argument("--labels", type=str, required=True, help="Path to the BIDS .tsv events file")
    parser_tune.add_argument("--epochs", type=int, default=2, help="Epochs per tuning run")
    parser_tune.set_defaults(func=handle_tune)

    # Command: serve
    parser_serve = subparsers.add_parser("serve", help="Spin up the real-time inference server")
    parser_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host binding")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port binding")
    parser_serve.add_argument("--actuation", type=str, choices=["simulated", "tcp"], default="simulated", help="Phase 7 Actuation Mode")
    parser_serve.set_defaults(func=handle_serve)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)

if __name__ == "__main__":
    main()
