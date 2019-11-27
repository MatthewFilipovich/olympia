from gym.envs.registration import register

register(
    id='olympia-rgb-v0',
    entry_point='olympia.envs:OlympiaRGB',
)

register(
    id='olympia-ram-v0',
    entry_point='olympia.envs:OlympiaRAM',
)
