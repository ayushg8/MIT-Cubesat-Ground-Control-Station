# agents/orchestrator.py — Mission Commander: dispatches specialists, synthesizes briefing
#
# This is the brain that manages the Ollama calls.
#   1. Gathers data from pipeline via get_analyzer_context()
#   2. Runs MissionAnalyzer to produce flag_report
#   3. Dispatches to relevant specialists
#   4. Makes one final Ollama call to synthesize a 3-sentence briefing
#   5. Logs all decisions to mission_state as PENDING_HUMAN_APPROVAL

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import config
from logic import MissionAnalyzer
from agents.specialists import (
    HazardReviewer,
    LandingAdvisor,
    SystemsHealth,
    _call_ollama,
    _parse_response,
)

logger = logging.getLogger(__name__)

_COMMANDER_PROMPT = """You are the Mission Commander for the MuraltZ CubeSat ground station. You have just received reports from three specialist officers. Your job is to synthesize their findings into a single briefing for the mission operator.

You must respond with EXACTLY three sentences:
- Sentence 1: Overall mission status (GO, CONDITIONAL GO, or NO-GO) and why in one line.
- Sentence 2: The single most critical item the operator must address right now.
- Sentence 3: The recommended next action.

Do not use bullet points. Do not use more than three sentences. Reference specific numbers from the specialist reports.

SPECIALIST REPORTS:
{specialist_reports}

FLAG SUMMARY:
{flag_summary}

YOUR 3-SENTENCE BRIEFING:"""


class MissionOrchestrator:
    """Coordinates MissionAnalyzer, three specialists, and synthesis."""

    def __init__(self):
        self._analyzer = MissionAnalyzer()
        self._hazard_reviewer = HazardReviewer()
        self._landing_advisor = LandingAdvisor()
        self._systems_health = SystemsHealth()
        self._last_briefing: dict | None = None
        self._run_count = 0

    def run_full_analysis(
        self,
        pipeline,
        mission_state,
        telemetry: dict,
    ) -> dict:
        """Run the complete advisor pipeline.

        Returns dict with: flag_report, specialist_results, briefing, decisions.
        """
        self._run_count += 1
        run_id = self._run_count

        # ── 1. Gather data from pipeline ──────────────────────────────────
        ctx = pipeline.get_analyzer_context()

        # ── 2. Run MissionAnalyzer ────────────────────────────────────────
        # Get landing candidates for cross-check
        try:
            landing_data = pipeline.recommend_landing_sites()
        except Exception as e:
            logger.warning(f"Orchestrator: landing recommendation failed: {e}")
            landing_data = {"candidates": []}

        flag_report = self._analyzer.analyze(
            observation_count=ctx["observation_count"],
            surveyed_mask=ctx["surveyed_mask"],
            fine_hazard_grid=ctx["fine_hazard_grid"],
            yolo_detections=ctx["yolo_detections"],
            fused_results=ctx["fused_results"],
            shadow_percentages=ctx["shadow_percentages"],
            image_metadata=ctx["image_metadata"],
            telemetry=telemetry,
            landing_candidates=landing_data.get("candidates", []),
        )

        logger.info(
            f"Orchestrator run {run_id}: {flag_report['summary']['total_flags']} flags, "
            f"status={flag_report['summary']['status']}"
        )

        # ── 3. Dispatch to specialists ────────────────────────────────────
        specialist_results = {}

        if self._hazard_reviewer.has_work(flag_report):
            logger.info(f"Orchestrator: dispatching HazardReviewer")
            specialist_results["hazard_reviewer"] = self._hazard_reviewer.analyze(
                flag_report,
                yolo_detector=getattr(pipeline, '_yolo_detector', None),
                image_dir=config.RECEIVED_DIR,
            )

        if self._landing_advisor.has_work(flag_report) or landing_data.get("candidates"):
            logger.info(f"Orchestrator: dispatching LandingAdvisor")
            coverage_pct = 0.0
            snapshot = mission_state.get_snapshot() if mission_state else {}
            coverage_pct = snapshot.get("coverage", {}).get("pct", 0.0)
            specialist_results["landing_advisor"] = self._landing_advisor.analyze(
                flag_report,
                landing_data=landing_data,
                coverage_pct=coverage_pct,
            )

        if self._systems_health.has_work(flag_report):
            logger.info(f"Orchestrator: dispatching SystemsHealth")
            specialist_results["systems_health"] = self._systems_health.analyze(
                flag_report,
                telemetry=telemetry,
            )

        # ── 4. Synthesize briefing ────────────────────────────────────────
        briefing = self._synthesize_briefing(flag_report, specialist_results)

        # ── 5. Build decision entries ─────────────────────────────────────
        decisions = self._build_decisions(
            run_id, flag_report, specialist_results, briefing
        )

        # Log decisions to mission_state
        if mission_state:
            for decision in decisions:
                mission_state.record_advisor_decision(decision)
            mission_state.save()

        self._last_briefing = briefing

        result = {
            "run_id": run_id,
            "flag_report": flag_report,
            "specialist_results": {
                k: self._sanitize_specialist_result(v)
                for k, v in specialist_results.items()
            },
            "briefing": briefing,
            "decisions": decisions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Orchestrator run {run_id} complete: "
            f"{len(specialist_results)} specialists called, "
            f"{len(decisions)} decisions logged"
        )

        return result

    def get_last_briefing(self) -> dict | None:
        """Return the most recent briefing (for lightweight polling)."""
        return self._last_briefing

    # ── Synthesis ─────────────────────────────────────────────────────────

    def _synthesize_briefing(
        self, flag_report: dict, specialist_results: dict
    ) -> dict:
        """Make one final Ollama call to produce the 3-sentence briefing."""

        # Build specialist summary text
        specialist_texts = []
        for agent_name, result in specialist_results.items():
            for resp in result.get("responses", []):
                specialist_texts.append(
                    f"[{agent_name.upper()}] "
                    f"Recommendation: {resp.get('recommendation', 'N/A')} | "
                    f"Confidence: {resp.get('confidence_level', 'N/A')} | "
                    f"Reasoning: {resp.get('reasoning', 'N/A')}"
                )

        if not specialist_texts:
            specialist_texts = ["No specialists were dispatched — no actionable flags detected."]

        flag_summary = json.dumps(flag_report["summary"], indent=2)
        specialist_report_text = "\n\n".join(specialist_texts)

        prompt = _COMMANDER_PROMPT.replace(
            "{specialist_reports}", specialist_report_text
        ).replace(
            "{flag_summary}", flag_summary
        )

        raw = _call_ollama(prompt)

        # Determine status from flag_report
        status = flag_report["summary"]["status"]
        pending_count = sum(
            1 for r in specialist_results.values()
            for resp in r.get("responses", [])
        )

        return {
            "text": raw.strip(),
            "status": status,
            "specialists_dispatched": list(specialist_results.keys()),
            "pending_decisions": pending_count,
            "total_flags": flag_report["summary"]["total_flags"],
        }

    # ── Decision building ─────────────────────────────────────────────────

    def _build_decisions(
        self,
        run_id: int,
        flag_report: dict,
        specialist_results: dict,
        briefing: dict,
    ) -> list[dict]:
        """Convert specialist responses into decision log entries."""
        decisions = []
        decision_idx = 0

        for agent_name, result in specialist_results.items():
            for resp in result.get("responses", []):
                decision_idx += 1
                flag = resp.get("flag", {})

                # Extract flag info
                if isinstance(flag, list):
                    flag_type = flag[0]["type"] if flag else "UNKNOWN"
                    severity = flag[0]["severity"] if flag else "INFO"
                    description = f"{len(flag)} flags"
                else:
                    flag_type = flag.get("type", "UNKNOWN")
                    severity = flag.get("severity", "INFO")
                    description = flag.get("context", {}).get("description", "")

                decisions.append({
                    "id": f"adv_{run_id:03d}_{decision_idx:02d}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id,
                    "agent": agent_name,
                    "flag_type": flag_type,
                    "severity": severity,
                    "description": description,
                    "ai_reasoning": resp.get("reasoning", ""),
                    "ai_recommendation": resp.get("recommendation", ""),
                    "ai_confidence": resp.get("confidence_level", ""),
                    "status": "PENDING_HUMAN_APPROVAL",
                    "operator_action": None,
                    "operator_note": None,
                    "resolved_at": None,
                })

        return decisions

    # ── Helpers ───────────────────────────────────────────────────────────

    def _sanitize_specialist_result(self, result: dict) -> dict:
        """Remove non-serializable data from specialist results."""
        sanitized = dict(result)
        for resp in sanitized.get("responses", []):
            # Remove numpy arrays or other non-serializable flag data
            flag = resp.get("flag")
            if flag is not None:
                if isinstance(flag, list):
                    resp["flag"] = [
                        {k: v for k, v in f.items()
                         if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                        for f in flag
                    ]
                elif isinstance(flag, dict):
                    resp["flag"] = {
                        k: v for k, v in flag.items()
                        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                    }
        return sanitized
