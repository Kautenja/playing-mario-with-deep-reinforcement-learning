DeepQAgent(
    env=<FrameStackEnv<PenalizeDeathEnv<DownsampleEnv<RewardCacheEnv<FrameSkipEnv<BinarySpaceToDiscreteSpaceEnv<TimeLimit<SuperMarioBrosEnv<SuperMarioBros-4-2-v0>>>>>>>>>,
    render_mode=None
    replay_memory_size=1000000,
    prioritized_experience_replay=False,
    discount_factor=0.99,
    update_frequency=4,
    optimizer=<keras.optimizers.Adam object at 0x7f4dc620edd8>,
    exploration_rate=AnnealingVariable(initial_value=1.0, final_value=0.1, steps=1000000),
    loss=huber_loss,
    target_update_freq=10000,
    dueling_network=False
)
