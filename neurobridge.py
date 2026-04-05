#!/usr/bin/env python3
"""
NeuroBridge — Neural Interface Bridge System

CLI entry point for training, evaluating, serving, and running
predictions with the NeuroBridge ECoG-to-Phoneme decoder.

Usage:
    python neurobridge.py train     [--epochs N] [--batch-size N]
    python neurobridge.py evaluate  [--samples N]
    python neurobridge.py serve     [--port N] [--debug]
    python neurobridge.py predict   [--top-k N]
    python neurobridge.py info
"""

import argparse
import sys
import logging

logger = logging.getLogger("neurobridge")


def cmd_train(args):
    """Train the NeuroBridge decoder model."""
    from neurobridge.train import train_model

    logger.info("Starting training...")
    history = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    final_loss = history.history["loss"][-1]
    final_acc = history.history["accuracy"][-1]
    print(f"\nTraining complete!")
    print(f"  Final loss:     {final_loss:.4f}")
    print(f"  Final accuracy: {final_acc:.4f}")


def cmd_evaluate(args):
    """Evaluate the trained model."""
    from neurobridge.evaluate import evaluate_model

    logger.info("Starting evaluation...")
    metrics = evaluate_model(num_test_samples=args.samples)

    print(f"\nEvaluation Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def cmd_serve(args):
    """Start the REST API server."""
    from neurobridge.api import start_server

    start_server(
        port=args.port,
        debug=args.debug,
    )


def cmd_predict(args):
    """Run a single prediction with mock data."""
    import numpy as np
    from neurobridge.config import Config
    from neurobridge.inference import RealtimeDecoder

    decoder = RealtimeDecoder()
    decoder.load()

    # Generate a mock ECoG frame
    mock_frame = np.random.rand(Config.NUM_FEATURES).astype(np.float32)

    top_k = decoder.predict_top_k(mock_frame, k=args.top_k)

    print(f"\nPrediction from mock ECoG frame ({Config.NUM_FEATURES} channels):")
    print(f"  Most likely phoneme: {top_k[0][0]} ({top_k[0][1]:.4f})")
    print(f"\n  Top {args.top_k} predictions:")
    for label, prob in top_k:
        bar = "█" * int(prob * 50)
        print(f"    {label:>6s}: {prob:.4f} {bar}")


def cmd_info(args):
    """Display configuration and system info."""
    from neurobridge.config import Config
    import neurobridge

    print(f"NeuroBridge v{neurobridge.__version__}")
    print()
    print(Config.summary())

    try:
        import tensorflow as tf
        print(f"\n  TensorFlow:     {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"  GPUs available: {len(gpus)}")
    except ImportError:
        print("\n  TensorFlow: NOT INSTALLED")


def main():
    parser = argparse.ArgumentParser(
        description="NeuroBridge — ECoG-to-Phoneme Neural Decoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    p_train = subparsers.add_parser("train", help="Train the decoder model")
    p_train.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    p_train.add_argument("--batch-size", type=int, default=None, help="Batch size")
    p_train.set_defaults(func=cmd_train)

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Evaluate the trained model")
    p_eval.add_argument("--samples", type=int, default=100, help="Test samples")
    p_eval.set_defaults(func=cmd_evaluate)

    # --- serve ---
    p_serve = subparsers.add_parser("serve", help="Start the REST API server")
    p_serve.add_argument("--port", type=int, default=None, help="API port")
    p_serve.add_argument("--debug", action="store_true", help="Enable debug mode")
    p_serve.set_defaults(func=cmd_serve)

    # --- predict ---
    p_pred = subparsers.add_parser("predict", help="Run a single prediction")
    p_pred.add_argument("--top-k", type=int, default=5, help="Top-K predictions")
    p_pred.set_defaults(func=cmd_predict)

    # --- info ---
    p_info = subparsers.add_parser("info", help="Show configuration info")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()