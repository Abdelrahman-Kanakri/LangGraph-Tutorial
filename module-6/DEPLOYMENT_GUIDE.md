# How to Build and Run This LangGraph Deployment

A step-by-step guide for building the Docker image and running this app
(API + Postgres + Redis) on your own machine.

---

## 0. The one rule that causes 90% of the problems

**Run every Docker / langgraph command from Konsole (the host terminal), NOT the
VS Code terminal.**

VS Code here is the Flatpak build. Its built-in terminal runs in a sandbox that
cannot see the host's Docker. From there you get the misleading error
`Docker not installed`. Konsole runs on the host, where Docker actually lives.

How to tell which terminal you're in — run `which docker`:

- prints `/usr/bin/docker`  -> you're on the host (Konsole). Good.
- prints nothing            -> you're in the VS Code sandbox. Open Konsole instead.

Edit code in VS Code. Run the stack in Konsole.

---

## 1. What each file in this folder is

| File | What it is | Do you edit it? |
|------|------------|-----------------|
| `langgraph.json`  | **The control file.** Tells langgraph which graph to serve, which Python version, which dependencies, and where the `.env` is. Everything starts here. | Rarely |
| `task_maistro.py` | **The app.** Defines the compiled graph. `langgraph.json` points at it as `./task_maistro.py:graph` (the `graph` variable in that file). | Yes — this is your logic |
| `configuration.py`| Runtime config schema for the assistant (model choice, user id, etc.). | Sometimes |
| `requirements.txt`| Python packages installed **into the Docker image** during build. | When you add a dependency |
| `.env`            | Secrets and API keys (LangSmith, model keys, etc.). **Not committed to git.** Referenced by `langgraph.json` via `"env": "../../.env"` (it lives at the repo root). | Yes — put your keys here |
| `DEPLOYMENT_GUIDE.md` | This guide. | No |

`langgraph.json` for this project looks like:

```json
{
    "dockerfile_lines": [],
    "graphs": { "task_maistro": "./task_maistro.py:graph" },
    "env": "../../.env",
    "python_version": "3.12",
    "dependencies": ["."]
}
```

Key line: `"env": "../../.env"` — this is how the server gets your keys. This langgraph
version has **no `--env-file` flag**; you configure the env file here instead.

---

## 2. One-time setup (only needed once per machine)

You already have these, but for a fresh machine:

1. **Docker** — engine + daemon running, and your user in the `docker` group.
   Check: `docker info` should succeed without sudo.
2. **Docker Compose** — required by `langgraph up`.
   Install on Nobara/Fedora: `sudo dnf install docker-compose`
   Check: `docker compose version`
3. **The project venv** — the `langgraph` CLI lives here, at the repo root `.venv`.
   Check: `ls ../../.venv/bin/langgraph`

---

## 3. Build and run — step by step

Open **Konsole**, then:

```bash
# 1. Go to the repo and activate the virtual environment
cd ~/Desktop/LangGraph-Tutorial
source .venv/bin/activate
# your prompt should now show (langgraph-tutorial) or (.venv)

# 2. Move into this deployment folder
cd module-6/deployment

# 3. (Optional) sanity-check the config
langgraph validate

# 4. Build the Docker image from this project
langgraph build -t my-image
#    - reads langgraph.json + requirements.txt
#    - produces an image named "my-image"

# 5. Run the full stack (API + Postgres + Redis)
langgraph up
#    - uses Docker Compose under the hood
#    - starts 3 containers and wires them together
#    - reads your keys from ../../.env (configured in langgraph.json)
```

> Do NOT use `docker run my-image` by itself. That starts ONLY the API container,
> with no Postgres and no Redis to connect to, so it crashes on startup.
> `langgraph up` is what brings the whole stack up together.

---

## 4. Confirm it's running

While `langgraph up` is running (it holds the terminal and streams logs):

- API docs:    open `http://localhost:8123/docs` in a browser
- Health check: open `http://localhost:8123/ok`

In a **second** Konsole tab you can also check the containers:

```bash
docker ps              # should list 3 running containers (api, postgres, redis)
docker volume ls       # the volume holding Postgres data
```

---

## 5. Stop it

- Foreground (started with `langgraph up`): press `Ctrl+C` in that terminal.
- If you started it detached (`langgraph up -d`): run `langgraph up` once to find it,
  or stop via Docker: `docker ps` then `docker stop <container-id>` for each.

---

## 6. Two ways to run the stack (FYI)

**Path A — `langgraph up` (recommended, what this guide uses).**
langgraph generates the Compose file for you, on the fly, and throws it away after.
Zero files to maintain. Best for normal development.

**Path B — your own `docker-compose.yml` (advanced).**
If you need to customize the stack (extra services, different ports, point Postgres
at your own/Supabase database), copy the example the tutorial ships
(`langchain-academy-main/module-6/deployment/docker-compose-example.yml`) to
`docker-compose.yml`, fill in real keys, then run `docker compose up`.
In that file the services find each other by name, e.g.
`REDIS_URI: redis://langgraph-redis:6379` and
`POSTGRES_URI: postgres://postgres:postgres@langgraph-postgres:5432/postgres`.
You can also feed your own file to langgraph: `langgraph up -d docker-compose.yml`.

There is no `langgraph compose` command. The closest "see what it generates" is
`langgraph dockerfile`, which prints the Dockerfile for the API image.

---

## 7. Troubleshooting (errors seen during setup)

| Error message | What it really means | Fix |
|---------------|----------------------|-----|
| `Error: Docker not installed` | You're in the VS Code Flatpak terminal; it can't see host Docker. | Run from Konsole. Verify with `which docker`. |
| `langgraph: command not found` | The venv isn't activated in this shell. | `source ~/Desktop/LangGraph-Tutorial/.venv/bin/activate` |
| `docker compose: unknown command` | Docker Compose isn't installed. | `sudo dnf install docker-compose` |
| `invalid env file ... contains whitespaces` | Only happens with `docker run --env-file`; Docker's parser rejects spaces around `=`. | Don't use `docker run --env-file`. Use `langgraph up`, which reads `.env` leniently. |
| Postgres `cannot parse 'bar'` / Redis `must specify scheme redis://` | You ran the bare API container with fake/placeholder DB URIs and no real services. | Use `langgraph up` — it provides real Postgres + Redis. |
| `No such option: --env-file` | This langgraph version configures the env file in `langgraph.json`, not via a flag. | Already set: `"env": "../../.env"` in `langgraph.json`. |
| `The "line" variable is not set` (warning) | Cosmetic Compose warning from the embedded Dockerfile's `for line in ...` loop. | Ignore it. Harmless. |

---

## Quick reference (the whole thing, once it's set up)

```bash
# In Konsole:
cd ~/Desktop/LangGraph-Tutorial && source .venv/bin/activate
cd module-6/deployment
langgraph build -t my-image     # only when code/deps change
langgraph up                    # start the stack  ->  http://localhost:8123/docs
# Ctrl+C to stop
```
