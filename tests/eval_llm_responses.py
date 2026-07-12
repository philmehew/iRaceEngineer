"""
LLM Response Evaluation — replay race snapshots and ask questions to log
the race engineer's responses for manual review.

Usage:
    python tests/eval_llm_responses.py [--count 50] [--output logs/eval.md]

Replays the Silverstone race data, picks snapshots at even intervals,
and asks a variety of race-engineering questions at each point.
Outputs a markdown file with prompt + response for review.
"""

import argparse
import json
import os
import sys

# Fix Windows console encoding for emoji/special characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from race_state import RaceState
from context_builder import ContextBuilder
from llm_client import LLMClient

# --- Questions to cycle through, designed to test different data areas ---

QUESTIONS = [
    # General strategy
    "What's my strategy for the next stint?",
    "How are we looking?",
    "Give me a status update.",
    # Fuel
    "Should I pit for fuel?",
    "How much fuel do I need to finish?",
    "Can I make it to the end on this fuel?",
    # Tyres
    "How are the tyres doing?",
    "Should I change tyres this stop?",
    "Are the tyres still ok?",
    # Damage / incidents
    "How much damage do I have?",
    "I had an incident, what should I do?",
    "Do I need a fast repair?",
    # Engine health
    "How's the engine?",
    "Is the oil temp ok?",
    "Should I be worried about engine temps?",
    # Pit strategy
    "When should I pit?",
    "What's my pit window?",
    "How many laps until I need to pit?",
    # Track / weather
    "What are the track conditions?",
    "Is it going to rain?",
    # Position / gaps
    "Where am I relative to the cars around me?",
    "Can I catch the car ahead?",
    "Am I safe from the car behind?",
    # Push-to-pass
    "Should I use push-to-pass?",
    # Brake / setup
    "What's my brake bias?",
    # Combined scenarios
    "I'm struggling with grip, what should I do?",
    "The car feels off, what would you check?",
    "What's the gap to the car ahead?",
    "",
    "Should I box this lap?",
    "How many laps of fuel left?",
    "What's my race engineer telling me?",
    "Tyre pressures ok?",
    "Any warnings I should know about?",
    "Are we on strategy?",
    "What's the plan for the next 10 laps?",
    "How's my pace compared to the leaders?",
]


def load_snapshots(session_dir: str, count: int):
    """Load evenly-spaced snapshots from a session directory."""
    all_files = sorted(
        f
        for f in os.listdir(session_dir)
        if f.startswith("snapshot_") and f.endswith(".json")
    )
    total = len(all_files)
    if total == 0:
        raise ValueError(f"No snapshots found in {session_dir}")

    # Pick evenly-spaced indices
    if count >= total:
        indices = list(range(total))
    else:
        indices = [int(i * (total - 1) / (count - 1)) for i in range(count)]

    snapshots = []
    for idx in indices:
        filepath = os.path.join(session_dir, all_files[idx])
        with open(filepath) as f:
            data = json.load(f)
        snapshots.append((idx + 1, data))  # snapshot_num is 1-based

    return snapshots


def load_all_snapshots(session_dir: str):
    """Load ALL snapshots from a session directory (in order)."""
    all_files = sorted(
        f
        for f in os.listdir(session_dir)
        if f.startswith("snapshot_") and f.endswith(".json")
    )
    snapshots = []
    for filepath_str in all_files:
        filepath = os.path.join(session_dir, filepath_str)
        with open(filepath) as f:
            data = json.load(f)
        snapshots.append(data)
    return snapshots


def check_response_quality(question: str, response: str, context: str) -> list[str]:
    """Check an LLM response for common quality issues.

    Returns a list of issue descriptions. Empty list means no issues found.
    """
    issues = []
    response_lower = response.lower()
    context_lower = context.lower()

    # 1. No refusal — the LLM must always provide useful info
    refusal_phrases = ["i'm busy", "try again", "ask again later", "cannot respond"]
    for phrase in refusal_phrases:
        if phrase in response_lower:
            issues.append(f"Refusal: contains '{phrase}'")
            break

    # 2. No pit recommendation when RACE ENDING SOON
    if "race ending soon" in context_lower or "!! race ending soon" in context_lower:
        pit_phrases = [
            "box this lap",
            "box lap",
            "pit now",
            "pit this lap",
            "should pit",
            "need to pit",
            "go to pit",
            "pitting is",
        ]
        for phrase in pit_phrases:
            if phrase in response_lower:
                issues.append(
                    "Bad pit advice: recommends pitting during RACE ENDING SOON"
                )
                break

    # 3. No tyre assessment when data is stale
    if "stale" in context_lower or "cannot assess condition" in context_lower:
        assessment_phrases = [
            "tyres are fine",
            "tyres look good",
            "pressures are good",
            "pressures stable",
            "pressures normal",
            "within normal range",
            "consider fresh",
            "if wear",
            "tyre temps are",
            "tyre wear is",
        ]
        for phrase in assessment_phrases:
            if phrase in response_lower:
                issues.append(f"Stale tyre assessment: '{phrase}' despite stale data")
                break

    # 4. No incident/damage confusion
    if "fast repair" in response_lower and "incident" in response_lower:
        # Fast repair fixes body damage, NOT incidents. If the response
        # connects fast repair to incidents without mentioning body/damage,
        # that's a confusion.
        if "damage" not in response_lower and "body" not in response_lower:
            issues.append(
                "Incident-damage confusion: mentions fast repair with incidents but no body damage context"
            )

    # 5. No "monitor" or "watch" language (one-shot advice rule)
    monitor_phrases = [
        "keep an eye on",
        "monitor the",
        "watch for",
        "watch the",
        "track the",
        "keep monitoring",
    ]
    for phrase in monitor_phrases:
        if phrase in response_lower:
            issues.append(f"Monitor language: '{phrase}' violates one-shot rule")
            break

    # 6. Highlight when engine warning present but not addressed in response
    if "engine warning" in context_lower:
        # Check if the response mentions the warning
        if (
            "engine warning" not in response_lower
            and "fuel pressure" not in response_lower
        ):
            issues.append("Engine warning in context but not highlighted in response")

    # 7. Check for wildly incorrect fuel calculations
    # If context shows fuel at < 5% and response says "laps of fuel" > 10,
    # that's a hallucinated calculation
    if "⚠ critical" in context_lower:
        # Extract any number like "X laps fuel" in response
        import re

        laps_match = re.search(r"(\d+)\+?\s*laps\s+(?:of\s+)?fuel", response_lower)
        if laps_match and int(laps_match.group(1)) > 10:
            issues.append(
                f"Wild fuel calc: claims {laps_match.group(1)} laps of fuel despite CRITICAL warning"
            )

    return issues


def run_eval(session_dir: str, count: int, output_file: str, config: dict):
    """Run the evaluation: load snapshots, ask questions, log responses.

    Feeds ALL snapshots into a single persistent RaceState so that
    lap history accumulates and fuel burn calculations work properly.
    Only queries the LLM at evenly-spaced intervals.
    """
    llm = LLMClient(config)
    context_builder = ContextBuilder(config)

    all_data = load_all_snapshots(session_dir)
    total_snaps = len(all_data)
    print(f"Loaded {total_snaps} snapshots from {session_dir}")

    # Pick evenly-spaced query indices into the full snapshot list
    if count >= total_snaps:
        query_indices = list(range(total_snaps))
    else:
        query_indices = [int(i * (total_snaps - 1) / (count - 1)) for i in range(count)]
    query_set = set(query_indices)

    # Team setup (done once, reused for every snapshot)
    team_config = config.get("team", {})
    teammate_list = team_config.get("teammates", [])
    driver_aliases = {}
    for entry in teammate_list:
        if isinstance(entry, str) and ":" in entry:
            iracing_name, real_name = entry.split(":", 1)
            driver_aliases[iracing_name.strip()] = real_name.strip()
        elif isinstance(entry, dict):
            for k, v in entry.items():
                driver_aliases[str(k)] = str(v)

    # Single persistent state — feed every snapshot so lap transitions
    # are detected and fuel_used per lap is recorded properly.
    state = RaceState(config)
    results = []
    question_idx = 0

    for snap_idx, data in enumerate(all_data):
        telemetry = data.get("telemetry", {})
        session_info = data.get("session_info", {})
        driver_names = data.get("driver_names", {})
        units = data.get("units", {})
        if driver_names:
            driver_names = {int(k): v for k, v in driver_names.items()}
        state.update(telemetry, session_info, driver_names, units=units)

        # Set team info on first snapshot (driver names available then)
        if snap_idx == 0 and driver_names:
            team_indices = set()
            for car_idx, name in driver_names.items():
                if name in driver_aliases:
                    team_indices.add(car_idx)
            state.set_team_indices(team_indices, driver_aliases)
            state.set_driver_names(driver_names)

        # Only query the LLM at selected intervals
        if snap_idx not in query_set:
            continue

        snapshot = state.get_snapshot()

        # Pick question (cycle through)
        question = QUESTIONS[question_idx % len(QUESTIONS)]
        question_idx += 1

        # Build prompt
        messages = context_builder.build_prompt(snapshot, question=question)

        # Call LLM
        try:
            response = llm.ask(messages)
        except Exception as e:
            response = f"[ERROR: {e}]"

        # Extract full prompt (system + user)
        system_prompt = messages[0]["content"]
        context = messages[1]["content"]

        # Summary line for the state
        player = snapshot.get("player", {})
        session = snapshot.get("session", {})
        summary = (
            f"Lap {player.get('lap', '?')} | "
            f"P{player.get('position', '?')} | "
            f"Fuel {player.get('fuel_pct', 0) * 100:.0f}% | "
            f"Flags: {', '.join(session.get('flags', ['Green']))}"
        )

        results.append(
            {
                "snapshot": snap_idx + 1,
                "question": question or "(general strategy)",
                "system_prompt": system_prompt,
                "context": context,
                "response": response or "(empty)",
                "state_summary": summary,
            }
        )

        done = len(results)
        print(f"  [{done}/{count}] Snap {snap_idx + 1}: {summary}")
        print(f"    Q: {question or '(general strategy)'}")
        print(f"    A: {(response or '(empty)')[:120]}...")
        print()

    # Write markdown report
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# LLM Response Evaluation\n\n")
        f.write(f"Session: `{session_dir}`\n")
        f.write(f"Model: `{config.get('llm', {}).get('model', 'unknown')}`\n")
        f.write(
            f"Context depth: `{config.get('prompt', {}).get('context_depth', 'full')}`\n"
        )
        f.write(f"Questions: {len(results)}\n\n")

        # Write system prompt once at the top (it's the same for all queries)
        if results and results[0].get("system_prompt"):
            f.write("## System Prompt\n\n")
            f.write(
                "<details>\n<summary>System prompt (sent with every query)</summary>\n\n"
            )
            f.write(f"```\n{results[0]['system_prompt']}\n```\n\n")
            f.write("</details>\n\n---\n\n")

        for r in results:
            f.write(f"## Snapshot {r['snapshot']} — {r['state_summary']}\n\n")
            q_display = r["question"] if r["question"] else "*(general strategy)*"
            f.write(f"**Q:** {q_display}\n\n")
            f.write("<details>\n<summary>Context sent to LLM</summary>\n\n")
            f.write(f"```\n{r['context']}\n```\n\n")
            f.write("</details>\n\n")
            f.write(f"**A:** {r['response']}\n\n")

            # Quality checks per response
            issues = check_response_quality(r["question"], r["response"], r["context"])
            if issues:
                f.write(f"**Issues:** {'; '.join(issues)}\n\n")
            else:
                f.write("**Quality:** ✅ OK\n\n")
            f.write("---\n\n")

        # Summary section
        f.write("## Quality Summary\n\n")
        total = len(results)
        results_with_issues = sum(
            1
            for r in results
            if check_response_quality(r["question"], r["response"], r["context"])
        )
        all_issues = []
        for r in results:
            all_issues.extend(
                check_response_quality(r["question"], r["response"], r["context"])
            )

        f.write(f"- Total responses: {total}\n")
        f.write(f"- Responses with issues: {results_with_issues}\n")
        f.write(f"- Clean responses: {total - results_with_issues}\n\n")

        # Count by issue type
        issue_counts: dict[str, int] = {}
        for issue in all_issues:
            issue_type = issue.split(":")[0] if ":" in issue else issue
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        if issue_counts:
            f.write("### Issue Breakdown\n\n")
            for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
                f.write(f"- {issue_type}: {count}\n")
            f.write("\n")

    print(f"\n✅ Evaluation complete. Results written to {output_file}")
    print(f"   {len(results)} questions asked.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM responses against race data"
    )
    parser.add_argument(
        "--session",
        default="tests/sample_data/session_2026-06-08_20-47-17",
        help="Path to session data directory",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of snapshots to evaluate (default: 50)",
    )
    parser.add_argument(
        "--output",
        default="logs/llm_eval.md",
        help="Output markdown file (default: logs/llm_eval.md)",
    )
    parser.add_argument(
        "--depth",
        default="full",
        choices=["minimal", "medium", "full"],
        help="Context depth (default: full)",
    )
    args = parser.parse_args()

    # Load config
    import yaml

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    # Override context depth
    config.setdefault("prompt", {})["context_depth"] = args.depth

    print("=" * 60)
    print("🏁 LLM Response Evaluation")
    print("=" * 60)
    print(f"Session: {args.session}")
    print(f"Questions: {args.count}")
    print(f"Output: {args.output}")
    print(f"Depth: {args.depth}")
    print(f"Model: {config.get('llm', {}).get('model', 'unknown')}")
    print()

    run_eval(args.session, args.count, args.output, config)


if __name__ == "__main__":
    main()
