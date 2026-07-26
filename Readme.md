# Secure Federated Learning with Differential Privacy 🔒🤝

A hands-on implementation of training a PyTorch model **without centralizing anyone's data**. Multiple simulated clients each train locally on their own slice of MNIST, share only model updates with a central server, and those updates are protected with differential privacy before they ever leave the client.

If you've never touched federated learning before, this repo is small enough to read end-to-end in one sitting — every piece maps to a single, short file.

> This project was built as the CIS\*6560 Cybersecurity summer project (2024) by **Ripunjoy Madhab Buddha** under Prof. Rozita Dara. The full write-up — motivation, methodology, and results — is in [`Summer project 2024 Final report .pdf`](./Summer%20project%202024%20Final%20report%20.pdf).

## Background & objective

Traditional centralized machine learning pools everyone's data into one place before training. That single store becomes a prime target for breaches and makes compliance with regulations like **GDPR** and **HIPAA** costly and complex.

The objective of this project is to develop and demonstrate a **secure federated learning framework that uses differential privacy (DP)** so multiple decentralized clients can jointly train one global model *without ever sharing their raw data* — preserving the privacy and confidentiality of sensitive information while still benefiting from the collective dataset.

### What it defends against

Combining federated learning (data never moves) with differential privacy (noise obscures any single record) hardens the model against several privacy attacks:

- **Membership inference** — determining whether a specific individual's data was in the training set. DP obscures each point's contribution to the model.
- **Model inversion** — reconstructing sensitive inputs from a trained model's outputs or parameters. DP noise plus decentralized data make reconstruction impractical.
- **Linkage attacks** — cross-referencing datasets to re-identify individuals. Keeping raw data on-device removes the datasets an attacker would need to correlate.

## Why this exists

Federated learning lets a model learn from data that never moves:

```
Client 0 (local MNIST shard)  ─┐
Client 1 (local MNIST shard)  ─┼──►  Server aggregates updates (FedAvg)  ──► Global model
Client 2 (local MNIST shard)  ─┘                                              │
        ▲                                                                     │
        └─────────────────────── updated weights broadcast back ─────────────┘
```

Each client trains on data it already owns, sends back only the model's *weights* (never the raw images), and the server averages those weights into a better global model. Run enough rounds and the global model converges roughly as if it had seen everyone's data — without anyone's data leaving their machine.

On top of that, this project layers **differential privacy** (via [Opacus](https://opacus.ai/)) on the client side, so even the shared weight updates carry a mathematical guarantee that no individual training example can be reverse-engineered from them.

## How the pieces fit together

| Stage | File | What happens |
|---|---|---|
| 1. Get data | [`dataset.py`](federated_tutorial/dataset.py) | Downloads MNIST via `torchvision` |
| 2. Partition data | [`lib/data.py`](federated_tutorial/lib/data.py) → `prepare_dataset()` | Splits the training set across `num_partitions` simulated clients (train/val loaders per client + one shared test loader) |
| 3. Define the model | [`model.py`](federated_tutorial/model.py) | A small CNN (`Net`) plus plain PyTorch `train()` / `test()` loops — no FL-specific code at all |
| 4. Start the server | [`server.py`](federated_tutorial/server.py) + [`lib/federated.py`](federated_tutorial/lib/federated.py) → `start_server()` | Spins up a [Flower](https://flower.ai/) server running `FedAvg`, aggregates client accuracy with a weighted average |
| 5. Add differential privacy | [`client.py`](federated_tutorial/client.py) via Opacus `PrivacyEngine` | Wraps the model/optimizer/dataloader so gradients are clipped and noised before training even starts |
| 6. Start a client | `lib/federated.py` → `flwr_client` / `start_client()` | Wraps your model + data into a Flower `NumPyClient` and connects it to the server |

The whole pipeline is deliberately decoupled: `model.py` has zero knowledge of Flower, and `lib/federated.py` has zero knowledge of MNIST or CNNs. Swap in your own PyTorch model and dataset and the rest keeps working.

## Try it yourself

### 1. Install dependencies

There's no `requirements.txt` yet, so grab the essentials directly:

```bash
pip install torch torchvision flwr opacus hydra-core numpy
```

### 2. Run the full simulation (server + 3 clients)

```bash
cd federated_tutorial
chmod +x run.sh   # first time only
./run.sh
```

This starts the server, waits a few seconds, then launches three clients in the background — each training on its own MNIST shard and reporting back over `localhost:5050`. Press `Ctrl+C` to stop everything cleanly.

### 3. …or run each piece by hand

```bash
cd federated_tutorial

# Terminal 1
python server.py

# Terminal 2, 3, 4
python client.py --client-id 0
python client.py --client-id 1
python client.py --client-id 2
```

Watching it as separate processes makes it easier to see what "federated" actually means — each client only ever prints its own local training/eval logs, never anyone else's data.

### 4. Check your GPU setup (optional)

```bash
python cuda.py
```

Prints whether PyTorch found a CUDA device — every script falls back to CPU automatically if not.

## Tuning knobs

These are the parameters you'll most likely want to play with, and where to find them:

| Parameter | Default | Where | What it controls |
|---|---|---|---|
| `num_partitions` | `3` | `client.py` → `prepare_dataset()` | How many simulated clients the training data is split into |
| `num_rounds` | `4` | `server.py` → `start_server()` | How many rounds of train → aggregate → broadcast the server runs |
| `epochs` | `15` | `client.py` → `fl_client(...)` | Local epochs each client trains per round |
| `noise_multiplier` | `1.1` | `client.py` → `privacy_engine.make_private()` | Higher = more privacy noise, lower model utility |
| `max_grad_norm` | `1.0` | `client.py` → `privacy_engine.make_private()` | Per-sample gradient clipping threshold before noise is added |
| `lr` / `momentum` | `0.1` / `0.9` | `client.py` → `torch.optim.SGD` | Local optimizer settings |

Turning `noise_multiplier` up and down is the fastest way to *feel* the privacy/accuracy trade-off — try re-running with `noise_multiplier=0.4` vs `2.0` and compare the accuracy each client reports.

## Repo layout

```
federated_tutorial/
├── run.sh              # boots 1 server + 3 clients, Ctrl+C to stop
├── server.py            # Hydra entrypoint that starts the Flower server
├── client.py             # entrypoint for a single client (--client-id N)
├── model.py              # CNN + plain PyTorch train/test loops
├── dataset.py            # MNIST download helper
├── cuda.py               # quick "do I have a GPU" check
└── lib/
    ├── federated.py     # Flower client/server glue (flwr_client, start_server, start_client)
    ├── data.py            # prepare_dataset(): splits data across clients
    └── diff-privacy.py   # standalone helper for wrapping a model with Opacus
```

> **Note:** there's a second copy of `lib/` at the repo root, identical to `federated_tutorial/lib/`. The runnable code lives under `federated_tutorial/` — treat the root copy as legacy/reference until it's cleaned up.

Hydra also expects a `conf/` directory for `server.py`'s configuration (it's git-ignored, and `server.py` currently runs with an empty config, so this is a good place to start if you want to make rounds/strategy configurable from the CLI instead of hardcoded).

## Reported results

The report evaluated a run with **10 clients over 4 rounds** on MNIST (10 classes, batch size 32, `lr=0.1`, `momentum=0.9`, 100 local epochs per client per round). Every client had to be online for both the fit and evaluate phases of each round.

| Round | Loss | Accuracy |
|---|---|---|
| 1 | 43.68 | 4.77% |
| 2 | — | 8.32% |
| 3 | — | 8.70% |
| 4 | 39.62 | 11.65% |

Loss fell steadily and accuracy rose across rounds, confirming the federated + DP loop trains correctly. Absolute accuracy stays modest — the differential-privacy noise, aggressive privacy settings, and limited compute all trade utility for privacy, so more rounds and tuning are needed for higher performance. The two biggest practical hurdles were **hardware/GPU limits** (FL is compute-hungry across many clients) and **sparse library documentation** (notably PySyft), which drove a lot of experimentation.

## Extending this

- **Swap the dataset**: replace `dataset.py` + the `Net` model in `model.py` — nothing in `lib/` assumes MNIST.
- **Swap the aggregation strategy**: pass a custom `strategy` into `start_server()` instead of the default `FedAvg`.
- **Add more clients**: bump `num_partitions` in `client.py` and add matching `python client.py --client-id N` calls to `run.sh`.
- **Simulate non-IID data**: adjust `partition_ratio` in `prepare_dataset()` to give clients unequal (or skewed) shares of the data.
