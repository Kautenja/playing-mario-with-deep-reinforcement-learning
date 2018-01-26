# python

Python code for quick automation tasks.

[`compile.py`](complie.py) compiles the LaTeX build folder by copying all
relevant files from the project hierarchy into a build directory. It relies on:

*   [`concatenate.py`](concatenate.py) concatenates Markdown in the
    [../md](../md) directory into a single markdown file, `build.md` in the
    build directory.

[`count_words.py`](count_words.py) is a standalone script to count the number
of words in the paper. It parses Markdown (with inline LaTeX) manually to
count the number of words in the text files in the [../md](../md) directory.
