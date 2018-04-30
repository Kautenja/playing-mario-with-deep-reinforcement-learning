# Playing Atari 2600 & Super Mario Bros. with Deep Reinforcement Learning

Using Double Dueling Deep-_Q_ Networks to play Atari 2600 games and Super
Mario Bros from the Nintendo Entertainment System (NES).

[![Playing Super Mario Bros with Deep Reinforcement Learning][thumb]][video]

[thumb]: https://img.youtube.com/vi/GCUVFAwUpj0/0.jpg
[video]: https://www.youtube.com/watch?v=GCUVFAwUpj0

# Installation

## `virtualenv`

Use `virtualenv` to contain the environment to a single
local installation of python3:

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

#### Test Cases

To execute the `unittest` suite for the project run:

```shell
make test
```

## Random Games

To play games with an agent that makes random decisions, use the random play
script.

```shell
python3 random_play.py <game name> <results directory>
```

-   `<game name>` is the game to play such as `Breakout` or `SuperMarioBros`
-   `<results directory>` is the directory to store output results in

## Train DDDQN

To train the DDDQN on a game, use the training script.

```shell
python3 dddqn_train.py <game name> <results directory>
```

-   `<game name>` is the game to play such as `Breakout` or `SuperMarioBros`
-   `<results directory>` is the directory to store output results in

## Play With Trained DDDQN

To run a trained DDDQN on validation games, use the play script.

```shell
python3 dddqn_play.py <results directory>
```

-   `<results directory>` is the directory containing a `weights.h5` file
    with stored weights from the dueling network model
