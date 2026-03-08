"""
agentic_pipeline.py – end-to-end runner for the agentic rare disease QA pipeline.

Usage (CLI):
    python pipelines/agentic_pipeline.py --query "What are the treatment options for KLA?"
    python pipelines/agentic_pipeline.py --interactive
    python pipelines/agentic_pipeline.py --eval --eval_file data/eval_questions.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.planning_agent import AgentResponse, PlanningAgent
from src.utils.config_loader import load_config
from src.utils.evaluation import AgentEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ────────────────────────────────────────────────────────────────────────────

def build_pipeline(config_path: str | None = None) -> PlanningAgent:
    """Construct the full agentic pipeline from configuration."""
    config = load_config(config_path)
    agent = PlanningAgent(config=config)
    logger.info("Pipeline ready.")
    return agent


# ────────────────────────────────────────────────────────────────────────────
# Run modes
# ────────────────────────────────────────────────────────────────────────────

def run_single_query(agent: PlanningAgent, query: str, verbose: bool = True) -> AgentResponse:
    """Run a single query through the pipeline and print results."""
    t0 = time.perf_counter()
    response = agent.run(query)
    elapsed = (time.perf_counter() - t0) * 1000

    if verbose:
        _print_response(response, elapsed)

    return response


def run_interactive(agent: PlanningAgent) -> None:
    """Start an interactive REPL session."""
    print("\n" + "=" * 60)
    print("  RareMind – Rare Disease AI Assistant")
    print("  Type 'quit' or 'exit' to stop | 'clear' to reset memory")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if query.lower() == "clear":
            agent.reset_memory()
            print("[Memory cleared – starting a new session]\n")
            continue

        response = agent.run(query)
        print(f"\nRareMind: {response.final_answer}\n")
        print(f"  [Route: {response.route} | Confidence: {response.confidence:.2f} | "
              f"Time: {response.total_duration_ms:.0f}ms]\n")


def run_evaluation(
    agent: PlanningAgent,
    eval_file: str,
    output_file: str = "./results/eval_results.json",
    verbose: bool = True,
) -> None:
    """
    Evaluate the pipeline on a JSON file of test questions.

    The eval file must be a JSON list of dicts with at least a "question" key.
    Optional keys: "reference_answer", "gold_route".
    """
    eval_path = Path(eval_file)
    if not eval_path.exists():
        logger.error("Eval file not found: '%s'.", eval_file)
        return

    with eval_path.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info("Running evaluation on %d questions…", len(questions))
    samples = []

    for i, item in enumerate(questions):
        query = item["question"]
        logger.info("Q%d/%d: %s", i + 1, len(questions), query[:80])

        t0 = time.perf_counter()
        response = agent.run(query)
        latency = (time.perf_counter() - t0) * 1000

        # Reset memory between eval questions for fair assessment
        agent.reset_memory()

        samples.append({
            "question": query,
            "answer": response.final_answer,
            "route": response.route,
            "evidence": {},  # evidence not stored in response; lightweight eval
            "reference_answer": item.get("reference_answer"),
            "gold_route": item.get("gold_route"),
            "latency_ms": latency,
        })

    evaluator = AgentEvaluator()
    summary = evaluator.evaluate_batch(samples)
    evaluator.save_results(summary, output_file)

    if verbose:
        print(f"\n{'=' * 50}")
        print("  Evaluation Summary")
        print(f"{'=' * 50}")
        print(f"  Samples:            {summary.n_samples}")
        print(f"  Faithfulness:       {summary.mean_faithfulness:.3f}")
        print(f"  Answer Relevancy:   {summary.mean_answer_relevancy:.3f}")
        print(f"  Response Safety:    {summary.mean_response_safety:.3f}")
        print(f"  Clarity:            {summary.mean_clarity:.3f}")
        print(f"  Overall:            {summary.mean_overall:.3f}")
        if summary.route_accuracy is not None:
            print(f"  Route Accuracy:     {summary.route_accuracy:.3f}")
        print(f"  Mean Latency:       {summary.mean_latency_ms:.0f} ms")
        print(f"  Results saved to:   {output_file}")
        print(f"{'=' * 50}\n")


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _print_response(response: AgentResponse, elapsed_ms: float) -> None:
    print("\n" + "─" * 60)
    print(f"Query:    {response.query}")
    print(f"Route:    {response.route}  |  Confidence: {response.confidence:.2f}")
    if response.disease_entities:
        print(f"Entities: {', '.join(response.disease_entities)}")
    print("─" * 60)
    print(f"\n{response.final_answer}\n")
    if response.sources:
        print("Sources:")
        for src in response.sources:
            print(f"  • [{src.get('type', '?')}] {src.get('label', '')}")
    print(f"\n⏱  {elapsed_ms:.0f} ms  |  Requery: {response.requery_count}x")
    print("─" * 60 + "\n")


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RareMind – Agentic Rare Disease QA Pipeline"
    )
    parser.add_argument("--query", "-q", type=str, help="Single query to answer.")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Start interactive session."
    )
    parser.add_argument(
        "--eval", action="store_true", help="Run evaluation mode."
    )
    parser.add_argument(
        "--eval_file", type=str, default="data/pseudo_dataset/eval_questions.json",
        help="Path to evaluation questions JSON file.",
    )
    parser.add_argument(
        "--output", type=str, default="results/eval_results.json",
        help="Evaluation output file path.",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.yaml."
    )
    parser.add_argument(
        "--trace", action="store_true", help="Print full agent reasoning trace."
    )
    args = parser.parse_args()

    agent = build_pipeline(args.config)

    if args.eval:
        run_evaluation(agent, args.eval_file, args.output)
    elif args.interactive:
        run_interactive(agent)
    elif args.query:
        response = run_single_query(agent, args.query)
        if args.trace:
            print("\n── Reasoning Trace ──")
            for step in response.trace:
                print(f"  Step {step.step} [{step.agent}] {step.action}")
                print(f"    → {step.result_summary} ({step.duration_ms:.0f}ms)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
