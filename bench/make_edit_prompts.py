"""Write edit_prompts.jsonl: 10 prompts that each embed a real Python file and ask for an edit.

  python make_edit_prompts.py [--lines 120]
"""
import argparse, json, os, sysconfig

TASKS = [
    ("bisect.py", "Add type hints to every function signature."),
    ("heapq.py", "Rename every parameter called `heap` to `h`."),
    ("textwrap.py", "Add a one-line comment above each method explaining what it does."),
    ("colorsys.py", "Replace the module-level constants with an Enum and update all uses."),
    ("fnmatch.py", "Add input validation that raises TypeError for non-string patterns."),
    ("shlex.py", "Add logging.debug calls at the start of each method."),
    ("glob.py", "Rename the function `glob1` to `glob_in_dir` everywhere."),
    ("stat.py", "Convert the first ten constants to use an IntFlag class."),
    ("keyword.py", "Add a function `is_reserved(s)` that checks both lists, and export it."),
    ("string.py", "Add docstrings to every function and method that lacks one."),
]
ap = argparse.ArgumentParser()
ap.add_argument("--lines", type=int, default=120, help="lines of each source file to include")
a = ap.parse_args()
lib = sysconfig.get_paths()["stdlib"]
with open("edit_prompts.jsonl", "w") as f:
    for name, task in TASKS:
        src = "".join(open(os.path.join(lib, name), encoding="utf-8").readlines()[: a.lines])
        prompt = f"{task} Return the complete updated file in one code block.\n\n```python\n{src}```"
        f.write(json.dumps({"prompt": prompt}) + "\n")
print("wrote edit_prompts.jsonl")
