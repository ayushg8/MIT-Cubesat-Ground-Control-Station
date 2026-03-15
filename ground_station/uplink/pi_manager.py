# uplink/pi_manager.py — Discover, connect to, and manage the Raspberry Pi CubeSat
#
# Uses mDNS (cubesat.local) for zero-config IP discovery, and paramiko for SSH.
# Called by dashboard API endpoints to give the operator one-click control.

import logging
import socket
import threading
import time

import paramiko

import config

logger = logging.getLogger(__name__)

# Pi credentials and paths
PI_HOSTNAME = "cubesat.local"
PI_USER = "cubesat"
PI_PASSWORD = "cubesat"
PI_SSH_PORT = 22
PI_FLIGHT_DIR = "/home/cubesat/MIT-BWSI-Cubesat/cubesat_flight"
PI_FLIGHT_CMD = "main.py"
# Log goes to the absolute path defined in the flight config.py (LOG_DIR)
PI_LOG_PATH = "/home/cubesat/cubesat_flight/data/logs/flight.log"

# SSH connection timeout
_SSH_TIMEOUT = 8


def discover() -> dict:
    """
    Discover the CubeSat Pi on the network.

    1. Try mDNS: resolve cubesat.local
    2. Test SSH reachability
    3. Test command port (5001) to see if flight software is running

    Returns dict with keys: found, ip, ssh_ok, flight_running, error
    """
    result = {
        "found": False,
        "ip": "",
        "ssh_ok": False,
        "flight_running": False,
        "error": "",
    }

    # Step 1: mDNS resolution
    ip = _resolve_mdns()
    if not ip:
        result["error"] = "Could not resolve cubesat.local — is the Pi on this WiFi network?"
        return result

    result["found"] = True
    result["ip"] = ip

    # Step 2: Test SSH
    result["ssh_ok"] = _test_ssh(ip)

    # Step 3: Test if flight software is running (command port open)
    result["flight_running"] = _test_command_port(ip)

    # Update config.CUBESAT_IP automatically
    config.CUBESAT_IP = ip
    logger.info(f"CubeSat discovered at {ip} (ssh={result['ssh_ok']}, flight={result['flight_running']})")

    return result


def start_flight_software() -> dict:
    """
    SSH into the Pi and start the flight software if not already running.

    Returns dict with keys: success, message, ip
    """
    ip = config.CUBESAT_IP or _resolve_mdns()
    if not ip:
        return {"success": False, "message": "CubeSat IP unknown — run discover first", "ip": ""}

    config.CUBESAT_IP = ip

    try:
        client = _ssh_connect(ip)
    except Exception as e:
        return {"success": False, "message": f"SSH failed: {e}", "ip": ip}

    try:
        # Check if already running
        _, stdout, _ = client.exec_command("pgrep -a python3 | grep main.py")
        running = stdout.read().decode().strip()

        if running:
            # Already running — just report
            pid = running.split()[0]
            client.close()
            return {"success": True, "message": f"Flight software already running (PID {pid})", "ip": ip}

        # Start the flight software in background
        cmd = (
            f"cd {PI_FLIGHT_DIR} && "
            f"nohup python3 {PI_FLIGHT_CMD} > /tmp/main_out.log 2>&1 &"
        )
        client.exec_command(cmd)
        time.sleep(2)  # give it a moment to start

        # Verify it started
        _, stdout, _ = client.exec_command("pgrep -a python3 | grep main.py")
        check = stdout.read().decode().strip()

        client.close()

        if check:
            pid = check.split()[0]
            logger.info(f"Flight software started on Pi (PID {pid})")
            return {"success": True, "message": f"Flight software started (PID {pid})", "ip": ip}
        else:
            # Grab last few lines of log for diagnostics
            return {"success": False, "message": "Process started but exited — check Pi logs", "ip": ip}

    except Exception as e:
        client.close()
        return {"success": False, "message": f"SSH command error: {e}", "ip": ip}


def stop_flight_software() -> dict:
    """SSH into the Pi and stop the flight software."""
    ip = config.CUBESAT_IP or _resolve_mdns()
    if not ip:
        return {"success": False, "message": "CubeSat IP unknown", "ip": ""}

    try:
        client = _ssh_connect(ip)
    except Exception as e:
        return {"success": False, "message": f"SSH failed: {e}", "ip": ip}

    try:
        client.exec_command("pkill -f 'python3 main.py'")
        time.sleep(1)

        # Verify it stopped
        _, stdout, _ = client.exec_command("pgrep -a python3 | grep main.py")
        check = stdout.read().decode().strip()
        client.close()

        if not check:
            logger.info("Flight software stopped on Pi")
            return {"success": True, "message": "Flight software stopped", "ip": ip}
        else:
            return {"success": False, "message": "Process still running after kill", "ip": ip}

    except Exception as e:
        client.close()
        return {"success": False, "message": f"Error: {e}", "ip": ip}


def get_pi_log(lines: int = 30) -> dict:
    """SSH into Pi and return the last N lines of the flight log."""
    ip = config.CUBESAT_IP or _resolve_mdns()
    if not ip:
        return {"success": False, "lines": [], "message": "CubeSat IP unknown"}

    try:
        client = _ssh_connect(ip)
        _, stdout, _ = client.exec_command(f"tail -n {lines} {PI_LOG_PATH} 2>/dev/null")
        output = stdout.read().decode("utf-8", errors="replace")
        client.close()
        return {"success": True, "lines": output.strip().split("\n") if output.strip() else []}
    except Exception as e:
        return {"success": False, "lines": [], "message": f"SSH failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_mdns() -> str:
    """Resolve cubesat.local via mDNS. Returns IP string or empty string."""
    try:
        results = socket.getaddrinfo(PI_HOSTNAME, None, socket.AF_INET)
        if results:
            ip = results[0][4][0]
            return ip
    except socket.gaierror:
        pass
    return ""


def _test_ssh(ip: str) -> bool:
    """Quick test: can we open an SSH connection?"""
    try:
        client = _ssh_connect(ip)
        client.close()
        return True
    except Exception:
        return False


def _test_command_port(ip: str) -> bool:
    """Test if the CubeSat flight software is listening on COMMAND_PORT."""
    try:
        with socket.create_connection((ip, config.COMMAND_PORT), timeout=2):
            return True
    except Exception:
        return False


def _ssh_connect(ip: str) -> paramiko.SSHClient:
    """Open an SSH connection to the Pi. Caller must close()."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        ip,
        port=PI_SSH_PORT,
        username=PI_USER,
        password=PI_PASSWORD,
        timeout=_SSH_TIMEOUT,
        allow_agent=False,
        look_for_keys=False,
    )
    return client
