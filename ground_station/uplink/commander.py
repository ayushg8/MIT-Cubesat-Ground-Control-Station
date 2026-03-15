# uplink/commander.py — Sends commands from GCS to CubeSat on COMMAND_PORT (5001)
#
# Each public method opens a fresh TCP connection, sends one JSON command + '\n',
# waits for a 1-byte ACK/NACK, then closes. This keeps the uplink stateless and
# avoids holding a persistent connection that the CubeSat's daemon thread would
# have to manage. The CubeSat's command_listener re-accepts after each command.
#
# All methods are safe to call from the Flask dashboard thread — connection errors
# are caught, logged, and returned as False so the UI can show a failure notice
# without crashing the server.

import json
import logging
import socket

import config
import protocol

logger = logging.getLogger(__name__)

# How long to wait for the CubeSat to ACK a command (seconds).
# Commands are processed quickly on the CubeSat — 5 s is generous.
_ACK_TIMEOUT_SEC = 5


class Commander:
    """Sends JSON commands to the CubeSat over TCP on COMMAND_PORT."""

    def __init__(self):
        self.last_error: str = ""   # Human-readable reason for last failure

    def send_command(self, cmd_dict: dict) -> bool:
        """
        Open a connection to the CubeSat, send JSON + newline, wait for ACK.

        Returns True on ACK, False on NACK or any connection/timeout error.
        Sets self.last_error with a human-readable reason on failure.
        CUBESAT_IP must be set in config before this is called.
        """
        self.last_error = ""

        if not config.CUBESAT_IP:
            self.last_error = "CUBESAT_IP not set"
            logger.error("CUBESAT_IP is not set in config — cannot send command")
            return False

        payload = (json.dumps(cmd_dict) + "\n").encode("utf-8")
        cmd_name = cmd_dict.get("cmd", "?")

        try:
            with socket.create_connection(
                (config.CUBESAT_IP, config.COMMAND_PORT),
                timeout=_ACK_TIMEOUT_SEC,
            ) as sock:
                sock.sendall(payload)
                logger.info(f"Command sent: {cmd_dict}")

                sock.settimeout(_ACK_TIMEOUT_SEC)
                response = sock.recv(1)

            if response == protocol.ACK:
                logger.info(f"ACK received for cmd='{cmd_name}'")
                return True
            elif response == protocol.NACK:
                self.last_error = "NACK — CubeSat rejected command"
                logger.warning(f"NACK received for cmd='{cmd_name}'")
                return False
            else:
                self.last_error = f"Unexpected response: {response!r}"
                logger.warning(
                    f"Unexpected response byte {response!r} for cmd='{cmd_name}'"
                )
                return False

        except ConnectionRefusedError:
            self.last_error = f"Connection refused ({config.CUBESAT_IP}:{config.COMMAND_PORT}) — is flight software running?"
            logger.error(f"Command '{cmd_name}' failed: {self.last_error}")
        except TimeoutError:
            self.last_error = f"Timeout waiting for ACK ({config.CUBESAT_IP}:{config.COMMAND_PORT})"
            logger.error(f"Command '{cmd_name}' timed out")
        except OSError as e:
            self.last_error = f"Network error: {e}"
            logger.error(f"Command '{cmd_name}' network error: {e}")

        return False

    # ------------------------------------------------------------------ #
    # Named command helpers                                                #
    # ------------------------------------------------------------------ #

    def retransmit(self, image_id: str) -> bool:
        """Ask the CubeSat to move image_id to the top of its downlink queue."""
        return self.send_command({"cmd": "retransmit", "image_id": image_id})

    def priority_cell(self, row: int, col: int) -> bool:
        """Boost novelty score for cell (row, col) so it gets imaged first."""
        return self.send_command({"cmd": "priority_cell", "row": row, "col": col})

    def set_cell(self, row: int, col: int) -> bool:
        """Override the CubeSat's current grid cell assignment."""
        return self.send_command({"cmd": "set_cell", "row": row, "col": col})

    def adjust_exposure(self, exposure_us: int) -> bool:
        """Set camera exposure time in microseconds for subsequent captures."""
        return self.send_command({"cmd": "adjust_exposure", "exposure_us": exposure_us})

    def enter_safe_mode(self) -> bool:
        """Command the CubeSat to enter SAFE_MODE immediately."""
        return self.send_command({"cmd": "enter_safe_mode"})

    def resume_normal(self) -> bool:
        """Exit SAFE_MODE and return the CubeSat to IDLE."""
        return self.send_command({"cmd": "resume_normal"})

    def request_status(self) -> bool:
        """Ask the CubeSat to send a telemetry packet immediately (out-of-band)."""
        return self.send_command({"cmd": "status_request"})

    def retry_downlink(self) -> bool:
        """Reset the CubeSat's consecutive-failure counter to resume downlink."""
        return self.send_command({"cmd": "retry_downlink"})

    def start_pass(self) -> bool:
        """Transition CubeSat from WAITING → IMAGING (begin a new imaging pass)."""
        return self.send_command({"cmd": "start_pass"})

    def end_pass(self) -> bool:
        """Transition CubeSat from IMAGING → PROCESSING (stop capture, begin downlink)."""
        return self.send_command({"cmd": "end_pass"})

    def set_grid_cell(self, row: int, col: int) -> bool:
        """Set the next grid cell to image in real time during a pass."""
        return self.send_command({"cmd": "cell", "row": row, "col": col})
