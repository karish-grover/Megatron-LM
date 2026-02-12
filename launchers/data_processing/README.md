# RedPajama Data Processing

Scripts for converting RedPajama raw dataset into Megatron binary format (.bin/.idx).

## Scripts

| Script | Description | Use Case |
|--------|-------------|----------|
| `preprocess_redpajama.sh` | Full dataset (all 10 sources) | Production training |
| `preprocess_redpajama_small.sh` | 2 sources (Wikipedia + StackExchange) | Development/testing |
| `preprocess_redpajama_tiny.sh` | 10k samples from Wikipedia | Quick debugging |
| ${\color{red}\textsf{[Karish]}}$ `download_redpajama.sh` | Parallel download of RedPajama-1T (~4.4 TB) | Data acquisition |
| ${\color{red}\textsf{[Karish]}}$ `prepare_tiny_pyxis.sh` | Download + preprocess 10k Wikipedia (Pyxis) | Quick test on Slurm clusters |
| ${\color{red}\textsf{[Karish]}}$ `prepare_redpajama_pyxis.sh` | Process all 10 sources inside Pyxis container | Full production processing on Slurm clusters |

## Usage

```bash
# Full dataset preprocessing
bash preprocess_redpajama.sh

# Small test dataset
bash preprocess_redpajama_small.sh

# Tiny dataset (configurable sample count)
NUM_SAMPLES=10000 bash preprocess_redpajama_tiny.sh
```

### ${\color{red}\textsf{[Karish]}}$ Pyxis/Enroot Cluster Workflow

For Slurm clusters with Pyxis/Enroot (no Docker/Apptainer needed):

```bash
# Step 1: Download full RedPajama (~4.4 TB, 16 parallel streams, ~14 hours)
nohup bash launchers/data_processing/download_redpajama.sh > /fsx/karish/data/download.log 2>&1 &

# Monitor progress:
tail -f /fsx/karish/data/download.log
du -sh /fsx/karish/data/redpajama-1t/*/

# Step 2: Process all 10 sources (requires salloc with GPU + 64 CPUs)
salloc --account=mrs_2 --qos=h200_mrs_2_high --time=24:00:00 \
       --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 \
       --cpus-per-task=64 --mem=0
bash launchers/data_processing/prepare_redpajama_pyxis.sh
```

**Tiny dataset (quick test):**

```bash
salloc --account=mrs_2 --qos=h200_mrs_2_high --time=2:00:00 \
       --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 \
       --cpus-per-task=10 --mem=0
bash launchers/data_processing/prepare_tiny_pyxis.sh
```

### ${\color{red}\textsf{[Karish]}}$ Download Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `PARALLEL` | 16 | Number of concurrent downloads |
| `SOURCES` | (all) | Filter to specific sources (e.g., `"wikipedia c4 arxiv"`) |
| `DATA_ROOT` | `/fsx/karish/data` | Root directory for all data |

### ${\color{red}\textsf{[Karish]}}$ Processing Configuration (Pyxis)

| Option | Default | Description |
|--------|---------|-------------|
| `TOKENIZER_MODEL` | `meta-llama/Llama-3.1-8B` | HuggingFace tokenizer (LLaMA for 1B HELM-MiCE) |
| `WORKERS` | 64 | Preprocessing worker processes |
| `PARTITIONS` | 16 | Data partitions (auto-adjusted per source) |
| `HF_TOKEN` | (unset) | HuggingFace token for gated models (required for LLaMA) |
| `SOURCES` | (all 10) | Filter to specific sources |

## Output

- `.bin` - Binary tokenized data
- `.idx` - Index file with document offsets
- `data_blend.txt` - Weight configuration for multi-source training
- ${\color{red}\textsf{[Karish]}}$ `data_blend_container.txt` - Same weights with container-internal paths (for Pyxis training scripts)

## Dependencies

These scripts call `tools/preprocess_data.py` (core Megatron utility) with HuggingFace tokenizer (default: Qwen/Qwen3-8B).

${\color{red}\textsf{[Karish]}}$ The Pyxis scripts additionally require:
- Slurm with Pyxis/Enroot plugin
- NGC PyTorch container (`nvcr.io#nvidia/pytorch:25.04-py3`)
- `transformers` + `zstandard` Python packages (auto-installed inside container if not using saved container)
