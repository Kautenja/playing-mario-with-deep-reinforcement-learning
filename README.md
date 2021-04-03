# Playing Super Mario Bros. With Deep Reinforcement Learning

[![Build Status](https://travis-ci.com/Kautenja/playing-mario-with-deep-reinforcement-learning.svg?branch=master)](https://travis-ci.com/Kautenja/playing-mario-with-deep-reinforcement-learning)

Using (Double/Dueling) Deep-Q Networks to play Super Mario Bros.

![DDQN-SMB-1-4](https://user-images.githubusercontent.com/2184469/113493396-8e6d3080-94a4-11eb-8e4c-956c277ac76f.gif)

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

The following instructions assume you have a shell running at the top level
directory of the project. For comprehensive documentation on command line
options, run the following:

```shell
python . -h
```

## Test Cases

To execute the `unittest` suite for the project run:

```shell
python -m unittest discover .
```

## Random Agent

To play games with an agent that makes random decisions:

```shell
python . -m random -e <environment ID>
```

-   `<environment ID>` is the ID of the environment to play randomly.

### Example

For instance, to play a random agent on Pong:

```shell
python . -m random -e Pong-v0
```

## Training A Deep-Q Agent

To train a Deep-Q agent to play a game:

```shell
python . -m train -e <environment ID>
```

-   `<environment ID>` is the ID of the environment to train on.

### Example

For instance, to train a Deep-Q agent on Pong:

```shell
python . -m train -e Pong-v0
```

## Playing With A Trained Agent

To run a trained Deep-Q agent on validation games:

```shell
python . -m play -o <results directory>
```

-   `<results directory>` is a directory containing a `weights.h5` file from a
    training session

### Example

For instance, to play a Deep-Q agent on Pong:

```shell
python . -m play -e results/Pong-v0/DeepQAgent/2018-06-07_09-24
```
