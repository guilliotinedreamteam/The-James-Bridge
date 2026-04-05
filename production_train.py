import logging
import os
import glob
import shutil
import subprocess
from neurobridge.cli import handle_train

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LeanOrchestrator:
    """
    Lean Production Phase Orchestrator.
    Implements a 'Streaming Enrollment' pattern: Download -> Train -> Finalize -> Purge.
    Prevents 'No space left on device' errors on systems with tight storage or inode limits.
    """
    def __init__(self, dataset_path: str, dataset_id: str = "ds003620"):
        self.dataset_path = dataset_path
        self.dataset_id = dataset_id
        self.passed = []
        self.failed = []

    def enroll_and_train(self, subject_id: str, epochs: int = 1):
        """Processes a single subject end-to-end then purges their raw data."""
        checkpoint_path = f"checkpoints/v1_{subject_id}_final.keras"
        if os.path.exists(checkpoint_path):
            logger.info(f"Weights already exist for {subject_id}. Skipping.")
            self.passed.append(subject_id)
            return

        # 1. Download specifically for this subject
        logger.info(f"--- Enrolling {subject_id} via S3 Streaming ---")
        sync_cmd = [
            "aws", "s3", "sync", 
            f"s3://openneuro.org/{self.dataset_id}", 
            self.dataset_path, 
            "--no-sign-request", "--exclude", "*", 
            "--include", f"{subject_id}/*"
        ]
        subprocess.run(sync_cmd, check=True)

        # 2. Resolve Paths
        # Pattern: datasets/openneuro_bids/ds003620/sub-XX/eeg/sub-XX_task-oddball_eeg.vhdr
        vhdr_pattern = os.path.join(self.dataset_path, subject_id, "eeg", "*_task-oddball_eeg.vhdr")
        headers = glob.glob(vhdr_pattern)
        if not headers:
            logger.error(f"Header not found for {subject_id}. Skipping.")
            self.failed.append((subject_id, "Header missing after sync"))
            return

        vhdr_path = headers[0]
        tsv_path = vhdr_path.replace("_eeg.vhdr", "_events.tsv")

        # 3. Train
        class Args: pass
        args = Args()
        args.file = vhdr_path
        args.labels = tsv_path
        args.epochs = epochs

        try:
            handle_train(args)
            best_path = "checkpoints/v1_offline_decoder_best.keras"
            if os.path.exists(best_path):
                os.rename(best_path, checkpoint_path)
            logger.info(f"Successfully finalized weights for {subject_id}.")
            self.passed.append(subject_id)
        except Exception as e:
            logger.error(f"Training failed for {subject_id}: {e}")
            self.failed.append((subject_id, str(e)))
        finally:
            # 4. Purge Subject Data to reclaim disk space
            subject_dir = os.path.join(self.dataset_path, subject_id)
            if os.path.exists(subject_dir):
                logger.info(f"Purging raw clinical data for {subject_id} to reclaim space.")
                shutil.rmtree(subject_dir)

    def run_full_enrollment(self, start_sub: int = 1, end_sub: int = 44, epochs: int = 1):
        for i in range(start_sub, end_sub + 1):
            sub_id = f"sub-{i:02d}"
            try:
                self.enroll_and_train(sub_id, epochs=epochs)
            except Exception as fatal:
                logger.critical(f"Orchestrator encountered fatal error on {sub_id}: {fatal}")

        logger.info("--- PRODUCTION COHORT SUMMARY (STREAMING) ---")
        logger.info(f"Enrollment Target: {end_sub - start_sub + 1}")
        logger.info(f"Successfully Passed: {len(self.passed)}")
        logger.info(f"Failed/Skipped: {len(self.failed)}")
        for sub, reason in self.failed:
            logger.warning(f"  - {sub}: {reason}")

if __name__ == "__main__":
    orchestrator = LeanOrchestrator("datasets/openneuro_bids/ds003620")
    # Start from sub-01 to ensure all naming conventions match, 
    # but the orchestrator will skip if checkpoints exist.
    orchestrator.run_full_enrollment(start_sub=1, end_sub=44, epochs=1)
