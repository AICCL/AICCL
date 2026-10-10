# AICCL

Code for **AICCL: Pushing the Quality-Scalability-Flexibility Frontier in Collective Communication via Learning** (NSDI).

AICCL uses a graph neural network and PPO to generate AllGather, AllReduce, AllToAll and AllToAll-v communication schedules. The supplied pretrained model is D512, using a hidden dimension of 64 and current-slot plan observations.

## Code structure

```text
.
├── rlccl/               # model, scheduling environment and PPO implementation
├── Data/
│   ├── train.json       # training topologies and traffic generation settings
│   └── test.json        # testing topologies and traffic generation settings
├── checkpoints/
│   └── D512.pth         # pretrained checkpoint
├── train.py             # PPO training and checkpoint recovery
├── test.py              # schedule generation with a trained checkpoint
├── requirements.txt
└── README.md
```

## Installation

Use Python 3.10–3.12 on Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch 2.5.1 is the reference version. For GPU training, install the corresponding CUDA-enabled PyTorch package for your environment.

## Training

```bash
python train.py --device cuda:0 --episodes 6000 --out checkpoints/my-model
```

Use `--device cpu` for CPU training. To try a short run, add `--until 16`; continue it with the same arguments plus `--resume`. Checkpoints and training metrics are saved under the output directory.

Training initializes a new model randomly. The supplied D512 checkpoint came from a historical warm start; a new training run is a separate model.

## Testing

Run one input for each collective on the four-node topology:

```bash
python test.py --device cpu --topology regular-4 --chunk-factor 1 \
  --limit 4 --out outputs/quick-test
```

The default checkpoint is `checkpoints/D512.pth`. To use your trained model, add `--checkpoint checkpoints/my-model/latest.pth`. Select a collective with `--collective allgather`, `allreduce`, `alltoall` or `alltoallv`.

Test all 1,992 supplied inputs:

```bash
python test.py --device cpu --out outputs/full-test
```

Results are saved as `results.jsonl`, `summary.json` and raw schedule `.npz` files. Use a new output directory for each test run. The default slot duration is 0.5 and the horizon is 40 slots; schedule time is measured in the communication capacity model.

## Data

Training and testing each use eight topology families: `regular-4`, `regular-8`, `regular-16`, `heterogeneous-3-5`, `heterogeneous-4-8`, `heterogeneous-4-4-8`, `sparse-ring-6` and `sparse-two-clique-8`. The JSON files store link capacities, shared resources and traffic generator states. Traffic inputs are generated deterministically at runtime, with chunk factors 1, 2 and 4.

Released under the [MIT license](LICENSE).
