<p align="center">
  <a href="https://github.com/satellite-acquisition/leopt"><img src="assets/leopt-logo.png" alt="LEOPT" width="360"></a>
</p>
<p align="center"><em>Bayesian search for satellite acquisition</em></p>
<p align="center">
  <a href="https://github.com/satellite-acquisition/leopt/actions/workflows/ci.yml"><img src="https://github.com/satellite-acquisition/leopt/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://github.com/satellite-acquisition/leopt/tree/v1.0.0"><img src="https://img.shields.io/badge/version-1.0.0-blue" alt="Version 1.0.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
  <a href="https://github.com/duncaneddy/brahe"><img src="https://img.shields.io/badge/Powered_by-Brahe-gold" alt="Powered by Brahe"></a>
</p>

# LEOPT

LEOPT is a Python library and web application for acquiring satellites when
their orbits are uncertain. It represents possible orbits as weighted particles,
plans where to point a ground antenna, and updates the search after detections
and misses.

The aim is to make acquisition methods easy to inspect, compare, and reuse.
The library includes finite-horizon beam search, continuation rollout, POMCP,
PFT-DPW, and baseline scan policies. You can use the search routines directly
with arrays of hypothesis weights, detection probabilities, and motion
constraints, or connect them to the orbit models and planning console.

**Powered by [Brahe](https://github.com/duncaneddy/brahe)** for independent SGP4
and ground-station geometry checks. The optional validation suite uses Brahe to
compute azimuth, elevation, and range, then compares those results with the
search engine's `python-sgp4` path.

## Quick Start

LEOPT requires Python 3.11 or newer. Install the library from source:

```bash
git clone https://github.com/satellite-acquisition/leopt.git
cd leopt
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For a small example, choose between two pointings and update the belief after a
miss. Each row of `detection` is a pointing; each column is an orbit hypothesis.

```python
import numpy as np
from antenna_pomdp.control.bayesian_search import greedy_action, posterior_after_miss

weights = np.array([0.2, 0.5, 0.3])
detection = np.array([[0.8, 0.2, 0.05], [0.05, 0.5, 0.8]])

action, probability = greedy_action(weights, detection)
posterior = posterior_after_miss(weights, detection[action])

print(f"Pointing {action}: detection probability {probability:.1%}")
print("Posterior after a miss:", posterior.round(3))
# Pointing 1: detection probability 50.0%
# Posterior after a miss: [0.38 0.5  0.12]
```

[examples/search.py](examples/search.py) extends this to several dwells with
antenna rate, acceleration, settling, and dwell constraints:

```bash
python examples/search.py
```

### Web Console

The console supports orbit ingest, station selection, belief visualization,
and schedule exports.

```bash
python -m pip install -e '.[platform]'
uvicorn acquisition_platform.service.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>; the API reference is at `/docs`. Set
`LEOPT_STATE_DB=/path/to/leopt.db` to retain sessions. Use one server worker,
since active sessions are held in process memory.

The console is a research prototype with demo and API-backed planning modes.
Schedules are advisory; the web service does not command hardware. Web plans
are capped at 20 dwells per pass, with incomplete tails identified in exports.
The browser loads libraries and fonts from public CDNs and needs an internet
connection.

## Going Further

| Source | What to explore |
| --- | --- |
| [Search and mount models](src/antenna_pomdp/control/) | Array-based Bayesian search, continuation rollout, and motion constraints |
| [POMDP planners](src/antenna_pomdp/pomdp/) and [baselines](src/antenna_pomdp/baselines/) | POMCP, PFT-DPW, raster, line, information-greedy, and Koopman policies |
| [Orbit geometry](src/antenna_pomdp/orbit/) and [belief models](src/antenna_pomdp/models/) | Propagation, uncertainty sampling, particle updates, and detection models |
| [Evaluation](src/antenna_pomdp/eval/) | Episode runners and schedule execution |
| [Application](src/acquisition_platform/) | FastAPI service, orbit ingest, console, and equipment interfaces |
| [Brahe validation](tests/test_brahe_truth.py) | Deterministic propagation and topocentric geometry comparisons |

Optional installs: `.[viz]` adds plots, `.[validation]` adds Brahe, and
`.[adapters]` adds serial and SNMP interfaces. The [Brahe documentation](https://docs.brahe.space/latest)
provides more on its astrodynamics tools.

## Development

With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --frozen --extra platform --extra viz --extra validation
uv run pytest -q
uv run ruff check .
```

CI checks Python 3.11–3.13, JavaScript syntax, the solver example, and installation
of the built wheel on every push and pull request. Bug reports and pull requests
are welcome through [GitHub](https://github.com/satellite-acquisition/leopt/issues).

## Versioning

Versions follow `MAJOR.MINOR.PATCH`: incompatible API changes, new features,
and fixes. Releases are tagged `vX.Y.Z`; the current release is `v1.0.0`.

## License

LEOPT is available under the [MIT License](LICENSE). Third-party dependencies
retain their own licenses.
