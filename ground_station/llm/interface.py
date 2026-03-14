# llm/interface.py — Local LLM interface via ollama subprocess
#
# Reads mission_state.json, injects it into the system prompt template, then
# calls `ollama run <model>` via subprocess with the prompt piped to stdin.
#
# This is the subprocess-based path. dashboard/app.py also has an independent
# urllib path that hits the ollama REST API directly — both work; they don't
# share state.
#
# Usage:
#   from llm.interface import query
#   response = query("Which grid cells have detected changes?")

import json
import logging
import os
import subprocess

import config

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROMPT_TEMPLATE_PATH = os.path.join(_HERE, "system_prompt.txt")

_OLLAMA_TIMEOUT_SEC = 30


class MissionInterface:
    """Wraps the ollama subprocess call with mission_state context."""

    def query(self, question: str) -> str:
        """
        Ask a question about the mission. Returns the model's response string.
        Never raises — errors are returned as human-readable strings.
        """
        mission_json = _load_mission_state()
        prompt = _build_prompt(question, mission_json)
        return _call_ollama(prompt)


def query(question: str) -> str:
    """Module-level convenience wrapper — same as MissionInterface().query()."""
    return MissionInterface().query(question)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_mission_state() -> str:
    """Load mission_state.json and return it as a formatted JSON string."""
    path = config.MISSION_STATE_FILE
    if not os.path.exists(path):
        logger.warning(f"LLM: mission_state.json not found at '{path}' — using empty state")
        return json.dumps({"note": "No mission data available yet."}, indent=2)
    try:
        with open(path) as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except Exception as e:
        logger.error(f"LLM: failed to read mission_state.json: {e}")
        return json.dumps({"error": f"Could not read mission state: {e}"}, indent=2)


def _load_prompt_template() -> str:
    """Read system_prompt.txt. Falls back to a hardcoded template if missing."""
    try:
        with open(_PROMPT_TEMPLATE_PATH) as f:
            return f.read()
    except Exception as e:
        logger.warning(f"LLM: could not read system_prompt.txt: {e} — using fallback")
        return (
            "You are a mission assistant. Answer using ONLY the data below. "
            "Do not invent information.\n\nMISSION DATA:\n{mission_state_json}"
        )


def _build_prompt(question: str, mission_json: str) -> str:
    """Substitute {mission_state_json} in the template and append the question."""
    template = _load_prompt_template()
    system_section = template.replace("{mission_state_json}", mission_json)
    return f"{system_section}\n\nQUESTION: {question}\n\nANSWER:"


def _call_ollama(prompt: str) -> str:
    """
    Run `ollama run <model>` with prompt piped to stdin.
    Returns the stdout string, or a descriptive error message.
    """
    model = config.LLM_MODEL
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=_OLLAMA_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            err = result.stderr.strip() or f"exit code {result.returncode}"
            logger.warning(f"LLM: ollama returned non-zero: {err}")
            return f"LLM error: {err}"

        response = result.stdout.strip()
        if not response:
            return "LLM returned an empty response."
        return response

    except FileNotFoundError:
        logger.warning("LLM: 'ollama' not found — is it installed? https://ollama.com")
        return "LLM unavailable: ollama is not installed. Install from https://ollama.com then run: ollama pull llama3.2"

    except subprocess.TimeoutExpired:
        logger.warning(f"LLM: query timed out after {_OLLAMA_TIMEOUT_SEC}s")
        return f"LLM query timed out ({_OLLAMA_TIMEOUT_SEC}s). The model may still be loading."

    except Exception as e:
        logger.error(f"LLM: unexpected error: {e}", exc_info=True)
        return f"LLM error: {e}"
