from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _run_research(query: str, hitl: bool = False) -> None:
    from energy_grid_research_agent.graph import get_app, make_initial_state
    from energy_grid_research_agent.report import ResearchReport

    app = get_app()
    initial = make_initial_state(query)
    config = {"configurable": {"thread_id": "cli-session"}}

    console.print(Panel(f"[bold cyan]Query:[/] {query}", title="Energy Grid Research Agent"))

    t0 = time.perf_counter()
    result = app.invoke(initial, config=config)
    latency = time.perf_counter() - t0

    report_dict = result.get("report") or {}
    report_dict.setdefault("query", query)
    report_dict.setdefault("summary", "No summary generated.")
    report_dict["latency_seconds"] = round(latency, 3)
    report_dict["agent_calls"] = result.get("agent_calls", 0)
    report_dict["human_approved"] = result.get("human_approved", False)
    report_dict["requires_human_review"] = result.get("requires_human_review", False)

    report = ResearchReport(**report_dict)

    console.print(Panel(report.summary, title="Summary"))

    if report.findings:
        table = Table(title="Findings", show_lines=True)
        table.add_column("Category", style="cyan")
        table.add_column("Description")
        table.add_column("Source")
        table.add_column("Conf.", justify="right")
        for f in report.findings:
            table.add_row(
                f.category,
                f.description[:80],
                f.source_document,
                f"{f.confidence:.2f}",
            )
        console.print(table)

    console.print(
        f"\n[dim]Agent calls: {report.agent_calls} | "
        f"Latency: {report.latency_seconds:.2f}s | "
        f"Validation: {'✓' if report.validation_passed else '✗'} | "
        f"HITL: {'approved' if report.human_approved else 'n/a'}[/]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy Grid Research Agent")
    parser.add_argument("--query", "-q", required=True, help="Research query")
    parser.add_argument("--hitl", action="store_true", help="Enable HITL mode")
    args = parser.parse_args()
    _run_research(args.query, hitl=args.hitl)


if __name__ == "__main__":
    main()
