from __future__ import annotations

import json
import urllib.error
import urllib.request

import config


def _mission_context(mission_state: dict, deterministic_briefing: dict | None = None) -> str:
    briefing = deterministic_briefing or mission_state.get("mission_briefing", {})
    compact = {
        "mission_briefing": briefing,
        "task_queue": mission_state.get("task_queue", [])[:5],
        "science_feed": mission_state.get("science_feed", [])[:8],
        "mission_metrics": mission_state.get("mission_metrics", {}),
        "coverage": mission_state.get("coverage", {}),
        "changes": mission_state.get("changes", {}),
        "route": mission_state.get("route", {}),
        "routes": mission_state.get("routes", {}),
        "cell_states": mission_state.get("cell_states", {}),
    }
    return json.dumps(compact, indent=2)


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
    state_json = _mission_context(mission_state)
    system_prompt = (
        "You are the MuraltZ mission operator copilot for a lunar terrain survey. "
        "Answer using only the provided mission context. "
        "Be short, operational, and decision-oriented. "
        "Prefer concrete actions, confidence/uncertainty notes, and what changed. "
        "If the answer is not supported by the data, say that directly."
        f"\n\nMISSION CONTEXT:\n{state_json}"
    )
    try:
        return _ollama_generate(system_prompt, question)
    except urllib.error.URLError as exc:
        return f"LLM unavailable: {exc}"
    except Exception as exc:
        return f"LLM unavailable: {exc}"


def generate_operator_briefing(mission_state: dict, deterministic_briefing: dict) -> str:
    state_json = _mission_context(mission_state, deterministic_briefing)
    briefing_json = json.dumps(deterministic_briefing, indent=2)
    system_prompt = (
        "You are producing an operator briefing for a lunar terrain survey mission. "
        "Use the deterministic briefing as the source of truth. "
        "Write like a mission controller, not a chatbot. "
        "State the mission phase, the top recommendation, why it matters now, and the risk of delay. "
        "Mention uncertainty or disagreement only when it affects action."
        f"\n\nDETERMINISTIC BRIEFING:\n{briefing_json}\n\nMISSION CONTEXT:\n{state_json}"
    )
    try:
        return _ollama_generate(
            system_prompt,
            "Write a 4-5 sentence operator briefing with these sections in prose: mission status, recommended action, why now, expected payoff."
        )
    except urllib.error.URLError as exc:
        return f"LLM unavailable: {exc}"
    except Exception as exc:
        return f"LLM unavailable: {exc}"
