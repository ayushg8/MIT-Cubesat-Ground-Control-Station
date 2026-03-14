# receiver/packet_handler.py — Validates incoming data from the CubeSat
#
# Checks:
#   1. Byte count matches declared file_size
#   2. MD5 of received bytes matches declared md5
#   3. Metadata dict is a valid JSON-serialisable object (already parsed by listener)
#
# Returns a ValidationResult so the caller (listener.py) decides what to do.

import hashlib
import json
from dataclasses import dataclass


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""       # Human-readable failure reason; empty on success
    computed_md5: str = "" # Always filled in (useful for logging even on failure)


def validate_transfer(data: bytes, declared_size: int, declared_md5: str, metadata: dict) -> ValidationResult:
    """
    Validate a received transfer.

    Args:
        data:          Raw bytes as received from the socket.
        declared_size: file_size value from the JSON header.
        declared_md5:  md5 value from the JSON header (hex string).
        metadata:      metadata dict already parsed from the header.

    Returns:
        ValidationResult with valid=True on success.
    """
    # 1. Size check
    if len(data) != declared_size:
        return ValidationResult(
            valid=False,
            reason=f"Size mismatch: received {len(data)} bytes, declared {declared_size}",
        )

    # 2. MD5 check
    computed = hashlib.md5(data).hexdigest()
    if computed != declared_md5:
        return ValidationResult(
            valid=False,
            reason=f"MD5 mismatch: computed {computed}, declared {declared_md5}",
            computed_md5=computed,
        )

    # 3. Metadata sanity check
    #    The listener already JSON-parsed the header so metadata is a dict.
    #    Verify it is re-serialisable (catches any non-JSON-safe values).
    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as e:
        return ValidationResult(
            valid=False,
            reason=f"Metadata not JSON-serialisable: {e}",
            computed_md5=computed,
        )

    return ValidationResult(valid=True, computed_md5=computed)
