# GPAE

This repository provides an implementation of **GPAE (Generalized Per-Agent Advantage Estimation)** for multi-agent policy optimization, built on top of **JaxMARL** and the JAX ecosystem.

Paper status: **Accepted at AAMAS 2026 (to appear).**

## Requirements

- Docker
- GNU Make

## Build

Build the Docker image:

```bash
make build
```

## Run

Run using the provided Makefile target:

```bash
make run
```

## Run (manual)

To run directly (e.g., inside the container), execute:

```bash
python baselines/MAPPO/gpae_rnn_smax.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kim2026gpae,
  title     = {Generalized Per-Agent Advantage Estimation for Multi-Agent Policy Optimization},
  author    = {Kim, Seongmin and Park, Giseung and Kim, Woojun and Jeon, Jiwon and Han, Seungyeol and Sung, Youngchul},
  booktitle = {Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year      = {2026}
}
```

## Acknowledgements

- Built on top of **JaxMARL**
- JAX ecosystem (e.g., JAX/Flax/Optax)
