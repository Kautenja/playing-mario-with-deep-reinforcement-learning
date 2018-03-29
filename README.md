# Evolutionary Search for Reinforcement Agents

<!-- Project Badges -->
[![build-status][]][build-server]

[build-status]: https://travis-ci.com/Kautenja/deep-learning-project.svg?token=FCkX2qMNHzx2qWEzZZMP&branch=master
[build-server]: https://travis-ci.com/Kautenja/deep-learning-project



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




## Project Proposal

The Markdown / LaTeX files comprising the proposal are in
[proposal](proposal). The corresponding [README](proposal/README.md)
outlines the structure.

### Usage

To compile the proposal into the [build](build) directory.

```shell
make proposal
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
[README](src/README.md) outlines the structure.

### Usage

#### Test Cases

To execute the unittest suite for the project run:

```shell
make test
```

This will recursively find test cases in [src](src) and run them.

#### Notebooks

Jupyter notebooks can be found in the [ipynb](ipynb) directory.
