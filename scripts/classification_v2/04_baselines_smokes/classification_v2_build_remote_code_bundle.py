"""Build or verify one canonical post-S1 remote runtime bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.remote_code_bundle import (
    build_remote_code_bundle,
    verify_remote_code_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--repository-root", required=True, type=Path)
    build.add_argument("--git-sha", required=True)
    build.add_argument("--output-dir", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--archive", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--git-sha", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_remote_code_bundle(
            repository_root=args.repository_root,
            requested_git_sha=args.git_sha,
            output_dir=args.output_dir,
        )
    else:
        result = verify_remote_code_bundle(
            archive_path=args.archive,
            manifest_path=args.manifest,
            expected_git_sha=args.git_sha,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
