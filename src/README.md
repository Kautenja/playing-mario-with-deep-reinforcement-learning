# Deprecated Legacy Source Tree

This directory contains the original Gym/Keras implementation. It is kept only
as historical reference while the PyTorch migration stabilizes.

The supported runtime package is `mario_rl`. Use the modern commands instead:

```shell
./main.sh config list
./main.sh train --config smb_dqn_fast_dev
./main.sh play --config smb_dqn_fast_dev
./main.sh random --config smb_dqn_fast_dev
```

Do not add new runtime code, tests, or dependencies under `src/`.
