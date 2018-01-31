# Deep Learning Project

## Project Proposal

The Markdown / LaTeX files comprising the proposal are in
[proposal](proposal). The corresponding [README](proposal/README.md)
outlines the structure.

### Usage

To compile the proposal into the [build](build) directory.

```shell
make proposal
```

## Literature Review

Papers reviewed by this work (and their citations) are in
[literature-review](literature-review). The corresponding
[README](literature-review/README.md) describes he structure.

### Usage

To update the `references.bib` for either to proposal or the paper:

```shell
make update_proposal_references
```

```shell
make update_paper_references
```

## Paper

The Markdown / LaTeX files comprising the paper are in
[paper](paper). The corresponding [README](paper/README.md)
outlines the structure.

### Usage

To compile the paper into the [build](build) directory.

```shell
make paper
```

## Source Code

Source code for the project lives in [src](src). The corresponding
[README](src/README) contains installation and execution instructions.
