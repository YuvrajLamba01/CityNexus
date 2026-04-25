---
title: CITYNEXUS — OpenEnv Server
emoji: "🏙️"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: OpenEnv FastAPI server for the CITYNEXUS multi-agent urban environment.
---

# CITYNEXUS — OpenEnv FastAPI Space

This Space hosts the **OpenEnv-compatible HTTP server** for [CITYNEXUS](https://github.com/YuvrajLamba01/CityNexus), a self-evolving multi-agent urban simulation built for the OpenEnv Hackathon 2026.

> **What is this Space for?** Judges and external clients can hit the standard OpenEnv routes (`/reset`, `/step`, `/state`, `/metadata`) over HTTP. The Space mirrors what `pip install` of the package would expose locally.
>
> **Looking for the playable demo?** That lives in a separate static Space — the in-browser city visualisation with live charts, agent message stream, and persistent memory.

## Endpoints

The standard OpenEnv `create_app(...)` HTTP surface is used:

| Method | Path        | Purpose                                                |
| ------ | ----------- | ------------------------------------------------------ |
| `POST` | `/reset`    | Reset to a fresh episode (`seed`, `episode_id`).       |
| `POST` | `/step`     | Step one tick with a `CityAction` (`mode`, `directive`). |
| `GET`  | `/state`    | Episode metadata (`episode_id`, `cumulative_reward`).  |
| `GET`  | `/metadata` | Action / observation schema + env info.                |

`CityAction.mode` is one of: `normal`, `emergency_focus`, `delivery_focus`, `defensive`.

The mode steers the Planner agent, which translates it into per-role priority directives that the four heuristic agents (Delivery, Traffic, Emergency, Police) react to inside the same tick.

## Environment configuration

All env vars are optional with sensible defaults:

| Var                          | Default | Meaning                                  |
| ---------------------------- | ------- | ---------------------------------------- |
| `CITYNEXUS_WIDTH`            | `20`    | Grid width                               |
| `CITYNEXUS_HEIGHT`           | `20`    | Grid height                              |
| `CITYNEXUS_MAX_TICKS`        | `100`   | Ticks per episode                        |
| `CITYNEXUS_DELIVERY_RATE`    | `0.30`  | Delivery spawn probability per tick      |
| `CITYNEXUS_INCIDENT_RATE`    | `0.10`  | Incident spawn probability per tick      |
| `CITYNEXUS_SEED`             | `42`    | Default RNG seed                         |
| `CITYNEXUS_MAX_CONCURRENT`   | `1`     | Max concurrent envs the Space will hold  |

## Local equivalent

```bash
git clone https://github.com/YuvrajLamba01/CityNexus
cd CityNexus
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then `curl localhost:8000/metadata`.

## Source & docs

* Repo: <https://github.com/YuvrajLamba01/CityNexus>
* Architecture, themes, training notebook: see the repository `README.md`.
* Training notebook (Unsloth + TRL GRPO): `notebooks/train_citynexus_colab.ipynb`.

## License

MIT.
