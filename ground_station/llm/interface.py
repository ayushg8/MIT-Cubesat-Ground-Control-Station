from __future__ import annotations

import json
import urllib.error
import urllib.request

import config


def _ollama_generate(system_prompt: str, prompt: str) -> str:
    payload = json.dumps({
        "model": config.LLM_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT_SEC) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("response", "No response from model").strip()


def query_mission(question: str, mission_state: dict) -> str:
    state_json = json.dumps(mission_state, indent=2)
    system_prompt = (
        "You are the MuraltZ mission operator assistant. "
        "Answer using only the provided mission state. "
        "If the answer is not supported by the data, say that directly. "
        "Keep answers short, concrete, and operational."
        f"\n\nMISSION STATE:\n{state_json}"
    )
    try:
        return _ollama_generate(system_prompt, question)
    except urllib.error.URLError as exc:
        return f"LLM unavailable: {exc}"
    except Exception as exc:
        return f"LLM unavailable: {exc}"


def generate_operator_briefing(mission_state: dict, deterministic_briefing: dict) -> str:
    state_json = json.dumps(mission_state, indent=2)
    briefing_json = json.dumps(deterministic_briefing, indent=2)
    system_prompt = (
        "You are producing a concise operator briefing for a lunar terrain survey mission. "
        "Use the deterministic briefing as the primary source of truth and only add detail "
        "that is supported by the mission state. Mention the top recommended task and why it matters."
        f"\n\nDETERMINISTIC BRIEFING:\n{briefing_json}\n\nMISSION STATE:\n{state_json}"
    )
    try:
        return _ollama_generate(system_prompt, "Write a 4-6 sentence operator briefing.")
    except urllib.error.URLError as exc:
        return f"LLM unavailable: {exc}"
    except Exception as exc:
        return f"LLM unavailable: {exc}"
