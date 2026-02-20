from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from neso_consultations.cache import SummaryCache
from neso_consultations.config import get_settings
from neso_consultations.llm.factory import build_llm_provider
from neso_consultations.models import dataclass_to_dict
from neso_consultations.service import ConsultationService


def build_service(*, require_llm: bool = True) -> ConsultationService:
    """Construct the application service with config, LLM provider, and cache.

    Input:
        require_llm: When `True`, wires the OpenAI provider; otherwise uses a
            no-op provider for read-only CLI commands.

    Output:
        Initialised `ConsultationService`.
    """
    settings = get_settings()
    llm_provider = build_llm_provider(settings, require_llm=require_llm)

    cache = SummaryCache(settings.cache_path)
    return ConsultationService(settings=settings, llm=llm_provider, cache=cache)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for UI launch, listing, and summary generation commands.

    Input:
        argv: Optional list of CLI args; defaults to process arguments.

    Output:
        Process-style exit code (`0` success, non-zero failure).
    """
    parser = argparse.ArgumentParser(description="Local consultation summarisation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ui_parser = subparsers.add_parser("ui", help="Launch Streamlit UI")
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", default="8501")

    org_parser = subparsers.add_parser("summary-org", help="Generate Approach 1 summary")
    org_parser.add_argument("--response-id", required=True)
    org_parser.add_argument("--no-cache", action="store_true")

    question_parser = subparsers.add_parser("summary-question", help="Generate Approach 2 summary")
    question_parser.add_argument("--question-id", required=True)
    question_parser.add_argument("--no-cache", action="store_true")

    list_org_parser = subparsers.add_parser("list-orgs", help="List available organisations")
    list_question_parser = subparsers.add_parser("list-questions", help="List available questions")

    args = parser.parse_args(argv)

    if args.command == "ui":
        return _launch_ui(host=args.host, port=str(args.port))

    try:
        service = build_service(require_llm=args.command in {"summary-org", "summary-question"})
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.command == "summary-org":
        result = service.summarise_organisation(response_id=args.response_id, use_cache=not args.no_cache)
        print(json.dumps(dataclass_to_dict(result), ensure_ascii=True, indent=2))
        return 0

    if args.command == "summary-question":
        result = service.summarise_question(question_id=args.question_id, use_cache=not args.no_cache)
        print(json.dumps(dataclass_to_dict(result), ensure_ascii=True, indent=2))
        return 0

    if args.command == "list-orgs":
        for response_id, label in service.list_organisations():
            print(f"{response_id}\t{label}")
        return 0

    if args.command == "list-questions":
        for question_id, label in service.list_questions():
            print(f"{question_id}\t{label}")
        return 0

    parser.print_help()
    return 1


def _launch_ui(*, host: str, port: str) -> int:
    """Start the Streamlit UI subprocess.

    Inputs:
        host: Bind address for the Streamlit server.
        port: Bind port for the Streamlit server.

    Output:
        Subprocess return code.
    """
    root_dir = Path(__file__).resolve().parents[1]
    app_path = root_dir / "neso_consultations" / "ui.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]

    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)
