import json
import logging
import socket
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProstheticInterface:
    """
    Phase 7: Actuation Interface.
    Bridges neural phoneme predictions to physical motor control protocols.
    Supports simulated actuation and TCP-based hardware communication.
    """

    def __init__(
        self,
        mode: str = "simulated",
        hardware_ip: str = "127.0.0.1",
        hardware_port: int = 9000,
    ):
        self.mode = mode
        self.hardware_ip = hardware_ip
        self.hardware_port = hardware_port

        # Mapping 41 phoneme indices to motor primitives.
        # This is a baseline mapping for prosthetic hand actuation.
        # In a clinical setting, this would be subject-specific and mapped to specific degrees of freedom.
        self.phoneme_to_motor = {
            0: "IDLE",
            1: "FINGER_INDEX_FLEX",
            2: "FINGER_MIDDLE_FLEX",
            3: "FINGER_RING_FLEX",
            4: "FINGER_PINKY_FLEX",
            5: "THUMB_OPPOSITION",
            6: "WRIST_PRONATION",
            7: "WRIST_SUPINATION",
            8: "GRASP_POWER",
            9: "GRASP_PRECISION",
            # Fill remaining to 40 with combinations or generic markers
        }
        # Initialize generic markers for the rest of the 41-class space
        for i in range(10, 41):
            self.phoneme_to_motor[i] = f"MOTOR_PRIMITIVE_{i}"

    def send_command(self, phoneme_id: int):
        """
        Translates a phoneme detection into a motor command and dispatches it to hardware.
        """
        action = self.phoneme_to_motor.get(phoneme_id, "IDLE")

        if self.mode == "simulated":
            logger.info(
                f"[SIMULATION] Predicted Phoneme {phoneme_id} -> Actuating {action}"
            )
            return True

        elif self.mode == "tcp":
            try:
                command = json.dumps(
                    {
                        "action": action,
                        "phoneme_id": phoneme_id,
                        "timestamp": time.time(),
                    }
                )
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.1)  # low latency limit
                    s.connect((self.hardware_ip, self.hardware_port))
                    s.sendall(command.encode("utf-8"))
                logger.info(f"[HARDWARE] TCP Command Sent: {action}")
                return True
            except Exception as e:
                logger.error(f"[HARDWARE] Failed to dispatch command: {e}")
                return False

        return False

    def batch_actuate(self, predictions: list):
        """
        Processes a sequence of phoneme predictions (e.g. from a sliding window)
        and filters for high-confidence transitions to prevent motor jitter.
        """
        for pid in predictions:
            if pid != 0:  # Skip idle
                self.send_command(pid)
