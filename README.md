# Playing Super Mario Bros. With Deep Reinforcement Learning

[![Build Status](https://travis-ci.com/Kautenja/playing-mario-with-deep-reinforcement-learning.svg?branch=master)](https://travis-ci.com/Kautenja/playing-mario-with-deep-reinforcement-learning)

Using deep reinforcement learning techniques to train agents to play Super
Mario Bros on the Nintendo Entertainment System (NES).

# Installation

## `virtualenv`

Use `virtualenv` to contain the Python environment to a single local
installation of python3:

#### Setup

To setup the virtual environment:

```shell
virtualenv -p python3 .env
source .env/bin/activate
```

When you've concluded the session:

```shell
deactivate
```

## Dependencies

[requirements.txt](requirements.txt) lists the Python dependencies for the
project with frozen versions. To install dependencies:

```shell
python -m pip install -r requirements.txt
```

**NOTE** if you're NOT using `virtualenv`, ensure that `python` aliases
python3; python2 is not supported.

# Usage

## Test Cases

To execute the `unittest` suite for the project run:

```shell
python -m unittest discover .
```

## Notebooks

To train agents, use notebooks in the `scripts` directory.

## Playing With A Trained Agent

To run a trained Deep-Q agent on validation games:

```shell
python . -o <results directory>
```

-   `<results directory>` is a directory containing a `weights.h5` file from a
    training session

### Example

For instance, to play a Deep-Q agent on Super Mario Bros. level 4-2:

```shell
python . -o results/SuperMarioBros-4-2-v0/DeepQAgent/TODO
```
