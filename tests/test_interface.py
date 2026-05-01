import json
import socket
from unittest.mock import MagicMock, patch

import pytest

from neurobridge.actuation.interface import ProstheticInterface


class TestProstheticInterface:
    def test_init_defaults(self):
        interface = ProstheticInterface()
        assert interface.mode == "simulated"
        assert interface.hardware_ip == "127.0.0.1"
        assert interface.hardware_port == 9000
        assert interface.phoneme_to_motor[0] == "IDLE"
        assert interface.phoneme_to_motor[1] == "FINGER_INDEX_FLEX"
        assert interface.phoneme_to_motor[40] == "MOTOR_PRIMITIVE_40"

    def test_simulated_mode(self):
        interface = ProstheticInterface(mode="simulated")

        # Test valid phoneme ID
        success = interface.send_command(1)
        assert success is True

        # Test unknown phoneme ID (should fallback to IDLE)
        success = interface.send_command(999)
        assert success is True

    @patch("socket.socket")
    def test_tcp_mode_success(self, mock_socket):
        interface = ProstheticInterface(mode="tcp")
        mock_sock_instance = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock_instance

        success = interface.send_command(2)

        assert success is True
        mock_socket.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_sock_instance.connect.assert_called_once_with(("127.0.0.1", 9000))
        mock_sock_instance.sendall.assert_called_once()

        # Verify JSON payload structure
        call_args = mock_sock_instance.sendall.call_args[0][0]
        payload = json.loads(call_args.decode("utf-8"))
        assert payload["action"] == "FINGER_MIDDLE_FLEX"
        assert payload["phoneme_id"] == 2
        assert "timestamp" in payload

    @patch("socket.socket")
    def test_tcp_mode_failure(self, mock_socket):
        interface = ProstheticInterface(mode="tcp")
        mock_sock_instance = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock_instance

        # Simulate a connection timeout/refusal
        mock_sock_instance.connect.side_effect = ConnectionRefusedError(
            "Connection refused"
        )

        success = interface.send_command(3)

        # Should gracefully catch the error and return False
        assert success is False

    def test_unsupported_mode(self):
        interface = ProstheticInterface(mode="unknown_mode")
        success = interface.send_command(1)
        assert success is False

    def test_batch_actuate(self):
        interface = ProstheticInterface(mode="simulated")

        with patch.object(interface, "send_command", return_value=True) as mock_send:
            # Predictions list containing IDLE (0) and active phonemes
            predictions = [0, 1, 0, 5, 0]
            interface.batch_actuate(predictions)

            # send_command should only be called for non-zero IDs
            assert mock_send.call_count == 2
            mock_send.assert_any_call(1)
            mock_send.assert_any_call(5)
