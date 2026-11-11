"""Word counts of the manuscript (abstract and main body by section). Needs pandoc on the PATH.

Usage: python tools/word_count.py <Manuscript_IJMI.docx>
"""
import re
import subprocess
import sys

text = subprocess.run(["pandoc", sys.argv[1], "-t", "plain", "--wrap=none"], capture_output=True, text=True,
                      check=True).stdout
count = lambda s: len(s.split())
abstract = text[text.index("Background:"):text.index("Keywords:")]
labels = len(re.findall(r"^(Background|Objective|Methods|Results|Conclusions):", abstract, re.M))
print(f"Abstract: {count(abstract) - labels} words (without the {labels} section labels)")
body = text[text.index("1. Introduction"):text.index("Data availability")]
starts = [body.index(s) for s in ["1. Introduction", "2. Methods", "3. Results", "4. Discussion", "5. Conclusions"]]
for i, j in zip(starts, starts[1:] + [len(body)]):
    print(f"{body[i:i + 16].strip():16s} {count(body[i:j])}")
print(f"Main body total: {count(body)} words (including section headings)")
