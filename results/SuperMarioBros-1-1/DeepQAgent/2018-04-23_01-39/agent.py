DeepQAgent(
    env=<Monitor<FrameStackEnv<ClipRewardEnv<PenalizeDeathEnv<DownsampleEnv<RewardCacheEnv<ToDiscreteWrapper<TimeLimit<SuperMarioBrosEnv instance>>>>>>>>>,
    render_mode='rgb_array'
    replay_memory_size=750000,
    discount_factor=0.99,
    update_frequency=4,
    optimizer=<keras.optimizers.Adam object at 0x7f17aecca518>,
    exploration_rate=AnnealingVariable(initial_value=0.10001, final_value=0.1, steps=1),
    loss=huber_loss,
    target_update_freq=10000,
    dueling_network=True
)