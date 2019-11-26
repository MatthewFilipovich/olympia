from gym.envs.registration import register

register(
    id='soccer-v0',
    entry_point='emergent_soccer.envs:FieldEnv',
)
