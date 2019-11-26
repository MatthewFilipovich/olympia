from gym.envs.registration import register

register(
    id='olympia-rgb',
    entry_point='olympia.envs:OlympiaRGB',
)

register(
    id='olympia-ram',
    entry_point='olympia.envs:OlympiaRAM',
)
