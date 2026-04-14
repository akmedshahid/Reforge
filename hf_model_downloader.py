#!/usr/bin/env python3
"""Advanced Hugging Face repository downloader.

Features:
- Accepts Hugging Face URLs (e.g., /org/model/tree/main)
- Interactive prompts (URL + destination) when flags are omitted
- Downloads full repository snapshots while preserving folder structure
- Organizes models under <base>/ai_models/<repo>/<revision>
- Supports include/exclude patterns, dry-runs, retries, and concurrency tuning
- Writes a JSON manifest with metadata and file inventory
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from huggingface_hub import HfApi, snapshot_download
except ModuleNotFoundError as import_error:
    HfApi = Any  # type: ignore[assignment]
    snapshot_download = None  # type: ignore[assignment]
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None

HF_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?huggingface\.co/(?P<repo>[^/]+/[^/]+)(?:/(?:tree|resolve)/(?P<revision>[^/?#]+))?",
    re.IGNORECASE,
)


@dataclass
class DownloadPlan:
    repo_id: str
    revision: str
    repo_type: str
    base_dir: str
    local_dir: str
    max_workers: int
    token_used: bool
    allow_patterns: list[str] | None
    ignore_patterns: list[str] | None
    estimated_bytes: int | None


def parse_hf_reference(reference: str, fallback_revision: str = "main") -> tuple[str, str]:
    """Parse a Hugging Face URL or repo_id into (repo_id, revision)."""
    ref = reference.strip()
    match = HF_URL_RE.match(ref)
    if match:
        repo_id = match.group("repo")
        revision = match.group("revision") or fallback_revision
        return repo_id, revision

    if re.match(r"^[^/\s]+/[^/\s]+$", ref):
        return ref, fallback_revision

    raise ValueError(
        "Invalid Hugging Face reference. Provide e.g. 'huggingface.co/org/model/tree/main' or 'org/model'."
    )


def csv_to_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts or None


def pretty_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def estimate_repo_size(api: HfApi, repo_id: str, revision: str, repo_type: str, token: str | None) -> int | None:
    """Estimate remote repository size by summing known sibling sizes."""
    info = api.repo_info(repo_id=repo_id, revision=revision, repo_type=repo_type, token=token)
    total = 0
    known = False
    for sibling in info.siblings or []:
        size = getattr(sibling, "size", None)
        if isinstance(size, int):
            total += size
            known = True
    return total if known else None


def build_plan(
    *,
    reference: str,
    revision: str | None,
    repo_type: str,
    destination_root: str,
    model_subdir: str,
    max_workers: int,
    token: str | None,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
) -> DownloadPlan:
    repo_id, parsed_revision = parse_hf_reference(reference, fallback_revision=revision or "main")
    chosen_revision = revision or parsed_revision

    safe_repo = repo_id.replace("/", "__")
    local_dir = Path(destination_root).expanduser().resolve() / model_subdir / safe_repo / chosen_revision

    api = HfApi(token=token)
    estimated_bytes = estimate_repo_size(
        api=api,
        repo_id=repo_id,
        revision=chosen_revision,
        repo_type=repo_type,
        token=token,
    )

    return DownloadPlan(
        repo_id=repo_id,
        revision=chosen_revision,
        repo_type=repo_type,
        base_dir=str(Path(destination_root).expanduser().resolve()),
        local_dir=str(local_dir),
        max_workers=max_workers,
        token_used=bool(token),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        estimated_bytes=estimated_bytes,
    )


def print_plan(plan: DownloadPlan) -> None:
    print("\n=== Download Plan ===")
    print(f"Repository      : {plan.repo_id}")
    print(f"Revision        : {plan.revision}")
    print(f"Repo type       : {plan.repo_type}")
    print(f"Destination     : {plan.local_dir}")
    print(f"Workers         : {plan.max_workers}")
    print(f"Token auth      : {'yes' if plan.token_used else 'no'}")
    print(f"Include patterns: {plan.allow_patterns or 'all files'}")
    print(f"Exclude patterns: {plan.ignore_patterns or 'none'}")
    print(f"Estimated size  : {pretty_size(plan.estimated_bytes)}")


def list_remote_files(api: HfApi, plan: DownloadPlan, token: str | None) -> list[str]:
    files = api.list_repo_files(
        repo_id=plan.repo_id,
        repo_type=plan.repo_type,
        revision=plan.revision,
        token=token,
    )
    return sorted(files)


def write_manifest(plan: DownloadPlan, files: Iterable[str], target: Path) -> Path:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": asdict(plan),
        "downloaded_files": list(files),
    }
    manifest_path = target / "download_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def perform_download(plan: DownloadPlan, token: str | None, retries: int, dry_run: bool) -> int:
    api = HfApi(token=token)
    remote_files = list_remote_files(api, plan, token)

    if dry_run:
        print("\n--- Dry Run (no files downloaded) ---")
        for item in remote_files:
            print(item)
        print(f"\nTotal files: {len(remote_files)}")
        return 0

    destination = Path(plan.local_dir)
    destination.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(
                repo_id=plan.repo_id,
                repo_type=plan.repo_type,
                revision=plan.revision,
                local_dir=plan.local_dir,
                local_dir_use_symlinks=False,
                max_workers=plan.max_workers,
                token=token,
                allow_patterns=plan.allow_patterns,
                ignore_patterns=plan.ignore_patterns,
                resume_download=True,
            )
            manifest_path = write_manifest(plan, remote_files, destination)
            print(f"\nDownload completed. Files: {len(remote_files)}")
            print(f"Manifest saved  : {manifest_path}")
            return 0
        except Exception as err:  # noqa: BLE001
            last_err = err
            print(f"Attempt {attempt}/{retries} failed: {err}")
            if attempt < retries:
                sleep_s = min(8 * attempt, 30)
                print(f"Retrying in {sleep_s}s...")
                time.sleep(sleep_s)

    print(f"Download failed after {retries} attempts: {last_err}")
    return 2


def interactive_prompt(args: argparse.Namespace) -> argparse.Namespace:
    if args.url:
        return args

    print("Advanced Hugging Face Downloader")
    print("Paste model URL or repo_id (example: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)")
    args.url = input("Model link/repo: ").strip()

    if not args.output:
        user_out = input("Base download folder (default: current folder): ").strip()
        args.output = user_out or os.getcwd()

    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download complete Hugging Face repositories with advanced controls.",
    )
    parser.add_argument("url", nargs="?", help="Hugging Face URL or repo_id (org/model)")
    parser.add_argument(
        "--revision",
        default=None,
        help="Git revision/branch/tag to download (default: revision from URL, otherwise main)",
    )
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"], help="Repository type")
    parser.add_argument("--output", default=None, help="Base output folder. Models are organized under <output>/ai_models/")
    parser.add_argument("--subdir", default="ai_models", help="Model root folder name under --output")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="Hugging Face token (or set HF_TOKEN)")
    parser.add_argument("--allow", default=None, help="Comma-separated include patterns (*.json,*.safetensors)")
    parser.add_argument("--ignore", default=None, help="Comma-separated exclude patterns (*.pt,*.bin)")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel worker threads")
    parser.add_argument("--retries", type=int, default=4, help="Download retries")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="List files only; do not download")
    return parser


def main() -> int:
    if _IMPORT_ERROR is not None:
        print("Missing dependency: huggingface_hub")
        print("Install it with one of the following commands:")
        print("  python -m pip install -r requirements.txt")
        print("  python -m pip install huggingface_hub")
        print(f"Technical detail: {_IMPORT_ERROR}")
        return 3

    parser = build_parser()
    args = parser.parse_args()
    args = interactive_prompt(args)

    output_root = args.output or os.getcwd()
    allow_patterns = csv_to_list(args.allow)
    ignore_patterns = csv_to_list(args.ignore)

    try:
        plan = build_plan(
            reference=args.url,
            revision=args.revision,
            repo_type=args.repo_type,
            destination_root=output_root,
            model_subdir=args.subdir,
            max_workers=max(1, args.max_workers),
            token=args.token,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    except Exception as err:  # noqa: BLE001
        print(f"Could not build plan: {err}")
        return 1

    print_plan(plan)

    if not args.yes:
        choice = input("Proceed? [y/N]: ").strip().lower()
        if choice not in {"y", "yes"}:
            print("Cancelled.")
            return 0

    return perform_download(plan, token=args.token, retries=max(1, args.retries), dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
