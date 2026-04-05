import argparse
import sys
import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_dataset(dataset_id: str, target_dir: str):
    """
    Downloads a legitimate clinical dataset from the OpenNeuro AWS S3 bucket.
    This bypasses the openneuro-cli authentication requirements for public datasets.
    Example: ds003620 (Speech decoding from ECoG)
    """
    target_path = os.path.join(target_dir, dataset_id)
    os.makedirs(target_path, exist_ok=True)
    
    logger.info(f"Initiating OpenNeuro AWS S3 sync for dataset: {dataset_id} into {target_path}")
    logger.info("This pulls directly from the public openneuro.org AWS S3 bucket.")
    
    try:
        # Check if aws cli is installed
        subprocess.run(["aws", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("AWS CLI is required to sync public datasets. Run: brew install awscli")
        sys.exit(1)
        
    try:
        # Execute the aws s3 sync command anonymously
        # Using --exclude "*" --include "sub-01/*" to grab only the first subject and prevent downloading
        # 100+ GB of data by default.
        cmd = [
            "aws", "s3", "sync", 
            f"s3://openneuro.org/{dataset_id}", 
            target_path, 
            "--no-sign-request", 
            "--exclude", "*", 
            "--include", "sub-01/*",
            "--include", "dataset_description.json",
            "--include", "participants.tsv"
        ]
        logger.info(f"Executing: {' '.join(cmd)}")
        
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully downloaded subject 'sub-01' of dataset {dataset_id} to {target_path}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"AWS S3 download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download clinical datasets from OpenNeuro S3")
    parser.add_argument("--dataset", type=str, required=True, help="OpenNeuro dataset ID (e.g., ds003620)")
    parser.add_argument("--target", type=str, default="datasets/openneuro_bids", help="Target directory")
    args = parser.parse_args()
    
    download_dataset(args.dataset, args.target)
