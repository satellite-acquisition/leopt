"""Check the installed wheel without importing source from the checkout."""

import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

from fastapi.testclient import TestClient

import acquisition_platform
import antenna_pomdp
from acquisition_platform.planner import schedule_export
from acquisition_platform.service.app import app


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    assert version("leopt") == antenna_pomdp.__version__ == expected
    assert schedule_export.__version__ == expected
    for package in (antenna_pomdp, acquisition_platform):
        installed_path = Path(package.__file__).resolve()
        assert installed_path.is_relative_to(Path(sys.prefix).resolve()), installed_path
        assert not installed_path.is_relative_to(root / "src"), installed_path

    with TestClient(app) as client:
        assert client.get("/openapi.json").json()["info"]["version"] == expected
        for path in (
            "/", "/console.css", "/intro.js", "/api.js", "/live.js", "/console.js",
            "/support.js", "/leopt-logo.png", "/api/networks",
        ):
            response = client.get(path)
            assert response.status_code == 200, (path, response.status_code)
            assert response.content, path
    print(f"LEOPT {expected}: installed wheel, API, and console assets passed")


if __name__ == "__main__":
    main()
