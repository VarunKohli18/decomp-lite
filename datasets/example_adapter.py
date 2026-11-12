"""Adapter template: plug in a dataset that is not a folder of .c files.

Copy to datasets/<your_name>.py and reference it as `- name: <your_name>` in a
function list. `load()` returns (or yields) one dict per sample:

    name      unique, filename-safe id; it names the outputs       (required)
    source    C source of one translation unit that compiles alone (required)
    function  symbol to decompile, when it is not `name`           (optional)
    cflags    extra compile flags for this sample                  (optional)

Anything goes inside: read a CSV, walk another repo, or pull from Hugging Face.
"""


def load():
    # e.g. a Hugging Face dataset whose rows carry a function name and its code:
    #
    #   from datasets import load_dataset
    #   for i, row in enumerate(load_dataset("org/name", split="test")):
    #       yield {"name": f"s{i:06d}", "function": row["fname"], "source": row["code"]}
    yield {
        "name": "abs_diff",
        "source": "int abs_diff(int a, int b)\n{\n    return a > b ? a - b : b - a;\n}\n",
    }
