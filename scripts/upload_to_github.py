"""Create a GitHub repo and upload this project without requiring git.exe.

Set GITHUB_TOKEN or GH_TOKEN before running. The token needs permission to
create repositories and write repository contents.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_REPO = "BEIRA-EEG-fMRI-Reconstruction"
DEFAULT_DESCRIPTION = "PyTorch autoencoder for reconstructing fMRI ROI signals from EEG time-series data."

EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".ipynb_checkpoints",
    "__pycache__",
    "Dataset",
    "data",
    "wandb",
    "figures",
    "plots",
    "outputs",
    "runs",
    "checkpoints",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".nii",
    ".set",
    ".mat",
}


def github_request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API error {exc.code} for {url}: {message}") from exc


def should_upload(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def collect_files(root: Path) -> list[Path]:
    files = [path for path in root.rglob("*") if should_upload(path, root)]
    return sorted(files, key=lambda item: item.as_posix().lower())


def create_or_get_repo(token: str, name: str, description: str, private: bool) -> dict:
    try:
        return github_request(
            "POST",
            "https://api.github.com/user/repos",
            token,
            {
                "name": name,
                "description": description,
                "private": private,
                "auto_init": False,
            },
        )
    except RuntimeError as exc:
        if "name already exists on this account" not in str(exc):
            raise
        user = github_request("GET", "https://api.github.com/user", token)
        return github_request("GET", f"https://api.github.com/repos/{user['login']}/{name}", token)


def try_github_request(method: str, url: str, token: str, payload: dict | None = None) -> dict | None:
    try:
        return github_request(method, url, token, payload)
    except RuntimeError as exc:
        if "GitHub API error 404" in str(exc):
            return None
        raise


def get_branch_state(token: str, full_name: str, branch: str) -> tuple[str | None, str | None]:
    ref = try_github_request(
        "GET",
        f"https://api.github.com/repos/{full_name}/git/ref/heads/{branch}",
        token,
    )
    if ref is None:
        return None, None

    parent_sha = ref["object"]["sha"]
    parent_commit = github_request(
        "GET",
        f"https://api.github.com/repos/{full_name}/git/commits/{parent_sha}",
        token,
    )
    return parent_sha, parent_commit["tree"]["sha"]


def upload_new_repo(root: Path, token: str, repo: dict, branch: str, message: str) -> str:
    owner = repo["owner"]["login"]
    full_name = repo["full_name"]
    parent_sha, base_tree_sha = get_branch_state(token, full_name, branch)
    files = collect_files(root)
    if not files:
        raise RuntimeError("No files matched the upload rules.")

    tree = []
    for path in files:
        relative_path = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        blob = github_request(
            "POST",
            f"https://api.github.com/repos/{full_name}/git/blobs",
            token,
            {
                "content": base64.b64encode(raw).decode("ascii"),
                "encoding": "base64",
            },
        )
        tree.append(
            {
                "path": relative_path,
                "mode": "100644",
                "type": "blob",
                "sha": blob["sha"],
            }
        )
        print(f"staged {relative_path}")

    tree_response = github_request(
        "POST",
        f"https://api.github.com/repos/{full_name}/git/trees",
        token,
        {"base_tree": base_tree_sha, "tree": tree} if base_tree_sha else {"tree": tree},
    )

    commit_payload = {
        "message": message,
        "tree": tree_response["sha"],
    }
    if parent_sha:
        commit_payload["parents"] = [parent_sha]

    commit = github_request(
        "POST",
        f"https://api.github.com/repos/{full_name}/git/commits",
        token,
        commit_payload,
    )

    if parent_sha:
        github_request(
            "PATCH",
            f"https://api.github.com/repos/{full_name}/git/refs/heads/{branch}",
            token,
            {
                "sha": commit["sha"],
                "force": False,
            },
        )
    else:
        github_request(
            "POST",
            f"https://api.github.com/repos/{full_name}/git/refs",
            token,
            {
                "ref": f"refs/heads/{branch}",
                "sha": commit["sha"],
            },
        )
    try:
        github_request(
            "PATCH",
            f"https://api.github.com/repos/{full_name}",
            token,
            {"default_branch": branch},
        )
    except RuntimeError:
        pass
    return f"https://github.com/{owner}/{repo['name']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-name", default=DEFAULT_REPO)
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--message", default="Update BEIRA project")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise SystemExit("Set GITHUB_TOKEN or GH_TOKEN before running this script.")

    root = Path(__file__).resolve().parents[1]
    repo = create_or_get_repo(token, args.repo_name, args.description, args.private)
    url = upload_new_repo(root, token, repo, args.branch, args.message)
    print(f"Uploaded project to {url}")


if __name__ == "__main__":
    main()
