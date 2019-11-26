from gym.envs.registration import register

register(
    id='soccer-RBG',
    entry_point='emergent_soccer.envs:FieldEnv',
)

register(
    id='soccer-RAM',
    entry_point='emergent_soccer.envs:FieldEnv',
)
