# Paper

## Outline

TODO

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine.

### Prerequisites

This application is dependency managed using [pip3][pip], the
[python3][python] package manager. It also relies on the Gnu tool [Make][].

#### MacOS

MacOS users will need to install [Homebrew][brew] before continuing with the
installation.

### Installation

The script can automatically install for your - Unix based - platform using:

```shell
make install
```

## Usage

### Authorship

This repository models a web design project. The [src][] directory contains a
hierarchy of subdirectories containing files of a certain filetype:

*   [src/bib][] - the folder with references files for the paper
*   [src/md][] - markdown files (chapters/sections)
    *   [src/md/img][] figures referenced by the markdown files
*   [src/python][] - Python code files
*   [src/sty][] - custom LaTeX styles that the paper uses (like the NIPS one)
*   [src/tex][] - LaTeX layout files
    *   this is where the central LaTeX layout file lives

### Compilation

These scripts compile all the markdown in [src/md][] in _alphanumeric_
order into any output files in [build][].

#### Markdown

This will concatenate all the LaTeX (.tex) and Markdown (.md) files in
[src/md][] into a single file [build/build.md][].

```shell
make markdown
```

#### LaTeX

This will first compile the master markdown file [build/build.md][], then use
[pandoc][] to parse the Markdown into LaTeX [build/build.tex][]. Note that
[pandoc][] supports inline LaTeX in the Markdown making Markdown a _superset_
of LaTeX.

```shell
make tex
```

#### PDF

This will first compile a LaTeX file [build/build.tex][], then use
[pdflatex][] and [bibtex][] to render the final product as a pdf
[build/build.pdf][].

```shell
make pdf
```

#### Word Count

This will render the printable pdf file [build/build.pdf][] then count the
words in it and print to the console.

```shell
make count_words
```

## Built With

*   [python3][python] - Scripting
*   [Make][] - Automation
*   [pandoc][] - Markdown parsing
*   [pdflatex][] - LaTeX to pdf conversion
*   [bibtex][] - Bibliography injection
*   [MacTeX][] - MacOS LaTeX support


<!-- Link shortcuts -->

<!-- Project files -->
[src]: ./src
[src/md]: ./src/md
[src/md/img]: ./src/md/img
[src/tex]: ./src/tex
[src/python]: ./src/python
[src/bib]: ./src/bib
[src/sty]: ./src/sty
<!-- Build Files -->
[build]: ./build
[build/build.md]: build/build.md
[build/build.tex]: build/build.tex
[build/build.pdf]: build/build.pdf
<!-- References -->
[Make]: https://www.gnu.org/software/make/
[brew]: https://brew.sh
[pip]: https://pip.pypa.io/en/stable/
[python]: https://www.python.org
[pandoc]: https://pandoc.org
[MacTeX]: http://www.tug.org/mactex/
[pdflatex]: https://www.tug.org/applications/pdftex/
[bibtex]: http://www.bibtex.org/
