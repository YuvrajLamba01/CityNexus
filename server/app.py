"""FastAPI app exposing CityNexusEnvironment over the OpenEnv HTTP protocol.

Run locally:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

The OpenEnv `create_app` helper wires reset / step / state / metadata routes
plus the web-interface playground.
"""

from __future__ import annotations

import os

from openenv.core.env_server import create_app

from server.environment import CityNexusEnvironment
from server.models import CityAction, CityObservation


def _make_env() -> CityNexusEnvironment:
    return CityNexusEnvironment(
        width=int(os.environ.get("CITYNEXUS_WIDTH", "20")),
        height=int(os.environ.get("CITYNEXUS_HEIGHT", "20")),
        max_ticks=int(os.environ.get("CITYNEXUS_MAX_TICKS", "100")),
        delivery_spawn_rate=float(os.environ.get("CITYNEXUS_DELIVERY_RATE", "0.30")),
        incident_spawn_rate=float(os.environ.get("CITYNEXUS_INCIDENT_RATE", "0.10")),
        default_seed=int(os.environ.get("CITYNEXUS_SEED", "42")),
    )


app = create_app(
    env=_make_env,
    action_cls=CityAction,
    observation_cls=CityObservation,
    env_name="citynexus_env",
    max_concurrent_envs=int(os.environ.get("CITYNEXUS_MAX_CONCURRENT", "1")),
)


def main() -> None:
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
