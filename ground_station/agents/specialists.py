# agents/specialists.py — Three specialist AI agents that consume MissionAnalyzer flags
#
# Each specialist:
#   1. Filters the flag_report for its relevant flag types
#   2. Gathers additional context data (images, detections, telemetry)
#   3. Builds a prompt from its system prompt template + flag data
#   4. Calls Ollama and parses the structured response
#   5. If it can't decide, it can re-examine raw data (YOLO re-detect, etc.)
#
# The orchestrator (built separately) manages calling these specialists
# and synthesizing their outputs.

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import urllib.error
from typing import Any

import config

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_TIMEOUT_SEC = 45


# ── Ollama call ───────────────────────────────────────────────────────────────

def _call_ollama(system_prompt: str, model: str | None = None) -> str:
    """Send a prompt to the Ollama REST API. Returns raw response text."""
    model = model or config.LLM_MODEL

    payload = json.dumps({
        "model": model,
        "prompt": "",  # everything is in the system prompt for single-turn
        "system": system_prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        _OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_OLLAMA_TIMEOUT_SEC) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "").strip()
    except urllib.error.URLError as e:
        logger.warning(f"Ollama connection failed: {e}")
        return f"LLM unavailable: cannot reach Ollama at {_OLLAMA_URL}. Is it running?"
    except TimeoutError:
        logger.warning(f"Ollama timed out after {_OLLAMA_TIMEOUT_SEC}s")
        return f"LLM query timed out ({_OLLAMA_TIMEOUT_SEC}s)."
    except Exception as e:
        logger.error(f"Ollama error: {e}", exc_info=True)
        return f"LLM error: {e}"


def _load_prompt(filename: str) -> str:
    """Load a system prompt template from agents/prompts/."""
    path = os.path.join(_PROMPTS_DIR, filename)
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {path}")
        return ""


def _parse_response(raw: str) -> dict:
    """Parse the three required fields from an agent response.

    Expects:
        REASONING: ...
        RECOMMENDATION: ...
        CONFIDENCE LEVEL: ...

    Returns dict with keys: reasoning, recommendation, confidence_level, raw.
    Gracefully handles missing fields.
    """
    result = {
        "reasoning": "",
        "recommendation": "",
        "confidence_level": "",
        "raw": raw,
    }

    # Try to extract each field with flexible matching
    reasoning_match = re.search(
        r"REASONING:\s*\n?(.*?)(?=\nRECOMMENDATION:|\Z)",
        raw, re.DOTALL | re.IGNORECASE,
    )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    rec_match = re.search(
        r"RECOMMENDATION:\s*\n?(.*?)(?=\nCONFIDENCE LEVEL:|\Z)",
        raw, re.DOTALL | re.IGNORECASE,
    )
    if rec_match:
        result["recommendation"] = rec_match.group(1).strip()

    conf_match = re.search(
        r"CONFIDENCE LEVEL:\s*\n?(.*)",
        raw, re.DOTALL | re.IGNORECASE,
    )
    if conf_match:
        result["confidence_level"] = conf_match.group(1).strip()

    return result


# ── Base class ────────────────────────────────────────────────────────────────

class BaseSpecialist:
    """Base class for specialist agents."""

    prompt_file: str = ""
    relevant_flag_types: list[str] = []

    def __init__(self):
        self._prompt_template = _load_prompt(self.prompt_file)

    def get_relevant_flags(self, flag_report: dict) -> list[dict]:
        """Filter flags to only those this specialist handles."""
        return [
            f for f in flag_report.get("flags", [])
            if f["type"] in self.relevant_flag_types
        ]

    def has_work(self, flag_report: dict) -> bool:
        """Return True if there are flags for this specialist to review."""
        return len(self.get_relevant_flags(flag_report)) > 0

    def analyze(self, flag_report: dict, **context) -> dict:
        """Run analysis. Override in subclasses to add context gathering."""
        raise NotImplementedError


# ── Hazard Reviewer ───────────────────────────────────────────────────────────

class HazardReviewer(BaseSpecialist):
    """Adversarial terrain analysis officer. Resolves DETECTION_CONFLICT flags.

    If the LLM can't decide (CONFIDENCE LEVEL: LOW), re-examines the source
    image through YOLO with a lower confidence threshold to gather more data.
    """

    prompt_file = "hazard_reviewer.txt"
    relevant_flag_types = ["DETECTION_CONFLICT"]

    def analyze(
        self,
        flag_report: dict,
        yolo_detector=None,
        image_dir: str | None = None,
    ) -> dict:
        """Analyze detection conflicts.

        Args:
            flag_report: from MissionAnalyzer.analyze()
            yolo_detector: optional YOLODetector instance for re-examination
            image_dir: path to received images (for re-examination)
        """
        flags = self.get_relevant_flags(flag_report)
        if not flags:
            return {
                "agent": "hazard_reviewer",
                "status": "no_flags",
                "responses": [],
            }

        responses = []

        for flag in flags:
            # Build prompt with this flag's data
            prompt = self._prompt_template.replace(
                "{flag_json}", json.dumps(flag, indent=2)
            )

            # Call Ollama
            raw_response = _call_ollama(prompt)
            parsed = _parse_response(raw_response)
            parsed["flag"] = flag

            # Re-examination: if LOW confidence and we have a YOLO detector,
            # re-run detection on the source image with lower threshold
            if (
                "LOW" in parsed["confidence_level"].upper()
                and yolo_detector is not None
                and image_dir is not None
            ):
                source_image = flag.get("context", {}).get("source_image", "")
                if source_image:
                    image_path = os.path.join(image_dir, source_image)
                    if os.path.exists(image_path):
                        logger.info(
                            f"HazardReviewer: re-examining {source_image} "
                            f"with lower threshold"
                        )
                        recheck_dets = yolo_detector.detect(
                            image_path, confidence_threshold=0.15
                        )
                        parsed["re_examination"] = {
                            "source_image": source_image,
                            "detections_at_015": [
                                {
                                    "class": d["class"],
                                    "confidence": d["confidence"],
                                    "center": d.get("center", []),
                                }
                                for d in recheck_dets
                            ],
                            "original_threshold": 0.20,
                            "recheck_threshold": 0.15,
                        }

                        # Second-pass: feed re-examination results back to LLM
                        followup_prompt = (
                            f"{prompt}\n\n"
                            f"ADDITIONAL DATA — RE-EXAMINATION:\n"
                            f"The source image was re-analyzed with a lower "
                            f"confidence threshold (0.15 vs 0.20).\n"
                            f"Results: {json.dumps(parsed['re_examination']['detections_at_015'], indent=2)}\n\n"
                            f"Given this additional data, update your assessment. "
                            f"Respond with REASONING, RECOMMENDATION, and CONFIDENCE LEVEL."
                        )
                        raw_followup = _call_ollama(followup_prompt)
                        parsed_followup = _parse_response(raw_followup)

                        # If the followup has higher confidence, use it
                        if "LOW" not in parsed_followup["confidence_level"].upper():
                            parsed["reasoning"] = parsed_followup["reasoning"]
                            parsed["recommendation"] = parsed_followup["recommendation"]
                            parsed["confidence_level"] = parsed_followup["confidence_level"]
                            parsed["re_examination"]["upgraded"] = True
                        else:
                            parsed["re_examination"]["upgraded"] = False

            responses.append(parsed)

        return {
            "agent": "hazard_reviewer",
            "status": "complete",
            "flags_reviewed": len(flags),
            "responses": responses,
        }


# ── Landing Advisor ───────────────────────────────────────────────────────────

class LandingAdvisor(BaseSpecialist):
    """Conservative landing safety lead. Enforces the Safe-Sample rule.

    Analyzes SURVEY_GAP flags in the context of landing candidates.
    Will not certify a zone with insufficient observation data regardless
    of the automated score.
    """

    prompt_file = "landing_advisor.txt"
    relevant_flag_types = ["SURVEY_GAP"]

    def analyze(
        self,
        flag_report: dict,
        landing_data: dict | None = None,
        coverage_pct: float = 0.0,
    ) -> dict:
        """Analyze survey gaps in context of landing zones.

        Args:
            flag_report: from MissionAnalyzer.analyze()
            landing_data: from pipeline.recommend_landing_sites()
            coverage_pct: current survey coverage percentage
        """
        flags = self.get_relevant_flags(flag_report)

        # Even with no SURVEY_GAP flags, if coverage is low, the landing
        # advisor should still weigh in on landing candidates
        if not flags and (landing_data is None or not landing_data.get("candidates")):
            return {
                "agent": "landing_advisor",
                "status": "no_flags",
                "responses": [],
            }

        # Build the landing context
        landing_json = json.dumps(
            {
                "candidates": landing_data.get("candidates", []) if landing_data else [],
                "coverage_pct": coverage_pct,
                "total_evaluated": landing_data.get("total_evaluated", 0) if landing_data else 0,
            },
            indent=2,
        )

        # Separate flags into landing-specific and general survey gaps
        landing_flags = [
            f for f in flags
            if f.get("context", {}).get("certification") == "DENIED"
        ]
        general_flags = [f for f in flags if f not in landing_flags]

        responses = []

        # Analyze landing-specific flags (one per candidate)
        for flag in landing_flags:
            prompt = self._prompt_template.replace(
                "{flag_json}", json.dumps(flag, indent=2)
            ).replace(
                "{landing_json}", landing_json
            )

            raw_response = _call_ollama(prompt)
            parsed = _parse_response(raw_response)
            parsed["flag"] = flag
            parsed["flag_category"] = "landing_zone"
            responses.append(parsed)

        # Analyze general survey gaps (batch them into one call)
        if general_flags:
            prompt = self._prompt_template.replace(
                "{flag_json}", json.dumps(general_flags, indent=2)
            ).replace(
                "{landing_json}", landing_json
            )

            raw_response = _call_ollama(prompt)
            parsed = _parse_response(raw_response)
            parsed["flag"] = general_flags
            parsed["flag_category"] = "general_coverage"
            responses.append(parsed)

        return {
            "agent": "landing_advisor",
            "status": "complete",
            "flags_reviewed": len(flags),
            "coverage_pct": coverage_pct,
            "responses": responses,
        }


# ── Systems Health ────────────────────────────────────────────────────────────

class SystemsHealth(BaseSpecialist):
    """Pragmatic flight engineer. Handles THERMAL_ALERT and SENSOR_DRIFT.

    Recommends specific operational changes — which systems to throttle,
    which software switches to flip.
    """

    prompt_file = "systems_health.txt"
    relevant_flag_types = ["THERMAL_ALERT", "SENSOR_DRIFT"]

    def analyze(
        self,
        flag_report: dict,
        telemetry: dict | None = None,
        telemetry_history: list[dict] | None = None,
    ) -> dict:
        """Analyze hardware/sensor flags.

        Args:
            flag_report: from MissionAnalyzer.analyze()
            telemetry: latest telemetry dict
            telemetry_history: list of recent telemetry readings (for trend)
        """
        flags = self.get_relevant_flags(flag_report)
        if not flags:
            return {
                "agent": "systems_health",
                "status": "no_flags",
                "responses": [],
            }

        # Build telemetry context
        telem_context = telemetry or {}

        # Add trend data if available
        if telemetry_history and len(telemetry_history) >= 2:
            temps = [
                t.get("cpu_temp_c")
                for t in telemetry_history
                if t.get("cpu_temp_c") is not None
            ]
            if len(temps) >= 2:
                telem_context = dict(telem_context)  # copy
                telem_context["_temp_trend"] = {
                    "readings": temps[-5:],
                    "rising": temps[-1] > temps[0],
                    "delta": round(temps[-1] - temps[0], 1),
                }

        telemetry_json = json.dumps(telem_context, indent=2)

        responses = []

        # Group thermal and drift flags separately — different analysis needed
        thermal_flags = [f for f in flags if f["type"] == "THERMAL_ALERT"]
        drift_flags = [f for f in flags if f["type"] == "SENSOR_DRIFT"]

        if thermal_flags:
            prompt = self._prompt_template.replace(
                "{flag_json}", json.dumps(thermal_flags, indent=2)
            ).replace(
                "{telemetry_json}", telemetry_json
            )

            raw_response = _call_ollama(prompt)
            parsed = _parse_response(raw_response)
            parsed["flag"] = thermal_flags
            parsed["flag_category"] = "thermal"
            responses.append(parsed)

        if drift_flags:
            prompt = self._prompt_template.replace(
                "{flag_json}", json.dumps(drift_flags, indent=2)
            ).replace(
                "{telemetry_json}", telemetry_json
            )

            raw_response = _call_ollama(prompt)
            parsed = _parse_response(raw_response)
            parsed["flag"] = drift_flags
            parsed["flag_category"] = "sensor_drift"
            responses.append(parsed)

        return {
            "agent": "systems_health",
            "status": "complete",
            "flags_reviewed": len(flags),
            "responses": responses,
        }


# ── Convenience: get all specialists ──────────────────────────────────────────

def get_all_specialists() -> dict[str, BaseSpecialist]:
    """Return a dict of all specialist instances keyed by name."""
    return {
        "hazard_reviewer": HazardReviewer(),
        "landing_advisor": LandingAdvisor(),
        "systems_health": SystemsHealth(),
    }
