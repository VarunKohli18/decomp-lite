# decomp-lite

A playground for **LLM-assisted decompilation with configurable pipelines**.

- **Pipelines are recipes you write**, not code: `input`, `recovery`, `reviewers`, `feedback`, `iterations`. The simplest one is a single LLM call.
- **Two inputs**: assembly (`objdump`) and pseudo-C (Ghidra), or both.
- **x86-64, every optimization level**: `O0 O1 O2 O3 Os`.
- **Bring your own dataset**: a folder of `.c` files, a JSONL file, or a ten-line Python adapter. ExeBench folders work as they are.
- **Three metrics**: does the answer compile (RC), does it compile against the code that calls it (RCI), does it pass the tests (RE).

```
config ─▶ dataset ─▶ build ─▶ asm / pseudo-C ─▶ pipeline ─▶ results/*.c ─▶ RC / RCI / RE
```

---

## Setup

Everything runs inside one Docker image (gcc, Ghidra, Python):

```bash
docker build --platform linux/amd64 -t decomp-lite .
docker run --platform linux/amd64 --rm -it --network host -v "$PWD":/work decomp-lite
```

Then tell it where your model is:

```bash
cp .env.example .env        # set your server address or API key
```

## How to use

Everything is set in `benchmark.yaml`. Go through it top to bottom, then run it.

```yaml
# Target: what the functions are compiled for.
archs: [x86_64]                  # x86_64
opts: [O0, O1, O2, O3, Os]       # any of: O0 | O1 | O2 | O3 | Os
compilers: [gcc]                 # gcc

# Model: the LLM behind every role.
model: vllm-gemma4-31b-it        # a name from models.yaml, or a list of names

# Functions: which datasets and functions to run.
functions_from: functions_example.yaml   # a file with a `repos:` list

# Pipelines: how an answer is produced. One entry per pipeline; the name is yours.
pipelines:
  single_pass:
    input: asm                   # asm | pseudoc | [asm, pseudoc]
    recovery: 1                  # how many LLMs write an answer: 1 or more, or a list of model names
    reviewers: 0                 # how many LLMs review it: 0 or more, or a list of model names
    feedback: none               # none | compiler | interface | tests
    iterations: 1                # 1 to 10

# Report: where the summary is written.
report:
  out_dir: reports/              # any folder
```

**1. Target.** Pick the optimization levels. The architecture is x86-64 and the compiler is gcc.

**2. Model.** Name a model from `models.yaml`. It has one example of each of the three kinds:

| Kind | `provider` | What to set in `.env` |
| --- | --- | --- |
| vLLM, a model you serve yourself | `vllm` | `VLLM_BASE_URL` |
| Ollama | `ollama` | `OLLAMA_BASE_URL` |
| API model: Claude or GPT | `anthropic` or `openai` | `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` |

To add a model, copy the entry of its kind and change `model`.

**3. Functions.** `functions_example.yaml` lists the dataset and the functions to run. See [Datasets](#datasets) to use your own.

**4. Pipeline.** The example above is the simplest one: one LLM, asked once. From there you can:

- change what the LLM sees: `input: pseudoc`, or both with `input: [asm, pseudoc]`
- add a reviewer that checks and corrects the answer: `reviewers: 1`
- let the LLM fix an answer that fails a check: `feedback: compiler` with `iterations: 2`
- get several answers and keep one that passes: `recovery: 3` with `feedback: compiler`

Add as many pipelines as you like under `pipelines:`, each with its own name. They run side by side and are compared in the report.

| `feedback` | The LLM is told |
| --- | --- |
| `none` | nothing |
| `compiler` | the compiler errors of its answer |
| `interface` | the errors of compiling its answer together with code that calls the function |
| `tests` | the above, plus which test cases fail |

`interface` and `tests` need a dataset with tests.

**5. Report.** Choose the folder for the summary.

**6. Run.**

```bash
python3 -m decomplite benchmark --config benchmark.yaml
```

Runs are resumable: finished work is skipped the next time. Add `--dry-run` to see what would run without running it.

## Results

| Where | What |
| --- | --- |
| `reports/summary.csv` | one row per function, opt and pipeline, with all metrics |
| `results/<pipeline>/<model>/.../<function>.c` | the final answer |
| `results/<pipeline>/<model>/.../<function>.json` | every prompt and reply behind it |
| `data/channels/.../asm.txt`, `pseudoc.c` | exactly what the LLM was shown |

| Metric | Column | Question |
| --- | --- | --- |
| RC | `standalone_compile` | Does the answer compile on its own? |
| RCI | `compile_ok` | Does it compile together with code that calls the function? |
| RE | `test_passed` | Does it then pass every test case? |

RCI and RE need a dataset with tests; otherwise they show `n/a`.

## Datasets

A dataset is a folder of C files in `datasets/`, one function per file, named after the function:

```
datasets/my_dataset/crc32.c          the function
datasets/my_dataset/crc32_test.c     optional test: includes crc32.c, calls it, exits 0 if correct
```

List it in the functions file:

```yaml
# Functions: one entry per dataset.
repos:
  - name: example                # dataset name: datasets/<name>/, <name>.jsonl or <name>.py
    path: datasets/example       # where the dataset is, if not under datasets/
    functions: [sum_array, count_bits, reverse_string, clamp_scale]   # exact names
    patterns: []                 # name patterns, e.g. ["sum_*"]; leave both empty to run everything
    cflags: []                   # extra compiler flags, e.g. ["-Iinclude"]
```

`datasets/example/` is a working example with tests. A dataset can also be a JSONL file (`datasets/example_jsonl.jsonl`) or a small Python loader (`datasets/example_adapter.py`). ExeBench folders work as they are.

## Notes

- The LLM only ever sees the stripped binary: the function's real name is never shown.
- Without Docker, on x86-64 Linux: install `gcc g++ binutils openjdk-21-jdk-headless`, run `pip install -e .` and `bash scripts/install_ghidra.sh`. Ghidra is only needed for `input: pseudoc`.
- Tests: `python3 -m pytest tests -q`.
