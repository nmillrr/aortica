"""Validate the full-stack Docker Compose environment (US-108).

These tests cover two layers:

* **Structural** — parse the compose / config files with PyYAML and assert the
  required services, volumes, health checks, and environment variables are
  present. These run everywhere (no Docker required).
* **Tooling** — when ``docker``/``docker compose`` and ``hadolint`` are available
  on the host, validate the compose configuration end-to-end and lint the
  Dockerfiles. These are skipped gracefully when the tools are absent (e.g. a
  minimal CI runner).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_COMPOSE = REPO_ROOT / "docker-compose.full.yml"
PROD_COMPOSE = REPO_ROOT / "docker-compose.prod.yml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
MAKEFILE = REPO_ROOT / "Makefile"
NGINX_CONF = REPO_ROOT / "deploy" / "nginx" / "nginx.conf"
PROXY_DOCKERFILE = REPO_ROOT / "deploy" / "nginx" / "Dockerfile.proxy"
QUICKSTART = REPO_ROOT / "docs" / "deployment" / "DOCKER_QUICKSTART.md"

DOCKERFILES = [
    REPO_ROOT / "Dockerfile.server",
    REPO_ROOT / "Dockerfile.edge",
    PROXY_DOCKERFILE,
]

REQUIRED_SERVICES = {"api", "frontend", "docs", "edge", "proxy"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_compose(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data: Dict[str, Any] = yaml.safe_load(fh)
    return data


def _docker_compose_cmd() -> Optional[List[str]]:
    """Return the docker compose invocation, or *None* if unavailable."""
    if shutil.which("docker"):
        # Prefer the v2 plugin form.
        try:
            subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                check=True,
                timeout=30,
            )
            return ["docker", "compose"]
        except (subprocess.SubprocessError, OSError):
            pass
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    return None


# ---------------------------------------------------------------------------
# File presence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        FULL_COMPOSE,
        PROD_COMPOSE,
        ENV_EXAMPLE,
        MAKEFILE,
        NGINX_CONF,
        PROXY_DOCKERFILE,
        QUICKSTART,
        *DOCKERFILES,
    ],
)
def test_deployment_files_exist(path: Path) -> None:
    """All deployment artifacts are present in the repository."""
    assert path.is_file(), f"Missing deployment file: {path}"


# ---------------------------------------------------------------------------
# Compose structure
# ---------------------------------------------------------------------------


def test_compose_defines_required_services() -> None:
    """The full compose file defines api, frontend, docs, edge, and proxy."""
    compose = _load_compose(FULL_COMPOSE)
    services = set(compose.get("services", {}))
    assert REQUIRED_SERVICES <= services, (
        f"Missing services: {REQUIRED_SERVICES - services}"
    )


def test_frontend_and_proxy_are_profile_gated() -> None:
    """frontend is dev-only and proxy is prod-only via compose profiles."""
    services = _load_compose(FULL_COMPOSE)["services"]
    assert services["frontend"].get("profiles") == ["dev"]
    assert services["proxy"].get("profiles") == ["prod"]


def test_all_services_have_healthchecks() -> None:
    """Every service defines a Docker health check."""
    services = _load_compose(FULL_COMPOSE)["services"]
    for name in REQUIRED_SERVICES:
        assert "healthcheck" in services[name], f"{name} has no healthcheck"
        assert "test" in services[name]["healthcheck"]


def test_shared_volumes_are_bind_mounted() -> None:
    """api mounts the shared ./data, ./models and ./logs host directories."""
    api = _load_compose(FULL_COMPOSE)["services"]["api"]
    mounts = {v.split(":")[0] for v in api.get("volumes", [])}
    for shared in ("./data", "./models", "./logs"):
        assert shared in mounts, f"api does not mount {shared}"


def test_service_dependency_ordering() -> None:
    """frontend/proxy start only after the api health check passes."""
    services = _load_compose(FULL_COMPOSE)["services"]
    frontend_dep = services["frontend"]["depends_on"]["api"]
    assert frontend_dep["condition"] == "service_healthy"
    proxy_dep = services["proxy"]["depends_on"]["api"]
    assert proxy_dep["condition"] == "service_healthy"


def test_prod_overlay_sets_production_env() -> None:
    """The prod overlay switches the api into production mode."""
    overlay = _load_compose(PROD_COMPOSE)["services"]
    assert overlay["api"]["environment"]["AORTICA_ENV"] == "production"


# ---------------------------------------------------------------------------
# .env.example
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "var",
    [
        "AORTICA_MODEL_PATH",
        "AORTICA_SECRET_KEY",
        "AORTICA_OAUTH_CLIENT_ID",
        "AORTICA_SYNC_URL",
        "AORTICA_LOG_LEVEL",
    ],
)
def test_env_example_documents_required_variables(var: str) -> None:
    """.env.example documents every variable named in the acceptance criteria."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert f"{var}=" in text, f"{var} not documented in .env.example"


# ---------------------------------------------------------------------------
# Makefile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["dev", "prod"])
def test_makefile_defines_targets(target: str) -> None:
    """The Makefile exposes the `dev` and `prod` targets."""
    text = MAKEFILE.read_text(encoding="utf-8")
    assert f"\n{target}:" in text, f"Makefile missing target: {target}"


def test_makefile_dev_uses_full_compose_file() -> None:
    """`make dev` runs against docker-compose.full.yml."""
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "docker-compose.full.yml" in text


# ---------------------------------------------------------------------------
# nginx reverse proxy
# ---------------------------------------------------------------------------


def test_nginx_terminates_tls_and_proxies() -> None:
    """nginx config terminates TLS and reverse-proxies the api and docs."""
    conf = NGINX_CONF.read_text(encoding="utf-8")
    assert "listen 443 ssl" in conf
    assert "ssl_certificate" in conf
    assert "proxy_pass http://api:8000" in conf
    assert "proxy_pass http://docs:8001" in conf


# ---------------------------------------------------------------------------
# Tooling: docker compose config validation
# ---------------------------------------------------------------------------


def test_docker_compose_config_validates() -> None:
    """`docker compose config` accepts the combined dev+prod configuration."""
    cmd = _docker_compose_cmd()
    if cmd is None:
        pytest.skip("docker compose not available on this host")

    result = subprocess.run(
        [
            *cmd,
            "-f",
            str(FULL_COMPOSE),
            "-f",
            str(PROD_COMPOSE),
            "--profile",
            "dev",
            "--profile",
            "prod",
            "config",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"docker compose config failed:\n{result.stderr}"
    )
    # The rendered config should list every service.
    for service in REQUIRED_SERVICES:
        assert service in result.stdout


# ---------------------------------------------------------------------------
# Tooling: hadolint Dockerfile linting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_dockerfiles_pass_hadolint(dockerfile: Path) -> None:
    """Dockerfiles lint cleanly (no error-level findings) under hadolint."""
    if shutil.which("hadolint") is None:
        pytest.skip("hadolint not available on this host")

    result = subprocess.run(
        ["hadolint", "--failure-threshold", "error", str(dockerfile)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"hadolint reported errors for {dockerfile.name}:\n{result.stdout}"
    )
