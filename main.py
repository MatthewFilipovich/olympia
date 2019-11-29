from olympia.envs import OlympiaRGB, OlympiaRAM
from olympia.envs.training_schemes import scheme

if __name__ == "__main__":
    # configure training values here
    shape = (15, 9)
    training_levels = list(scheme.keys())
    batch_size = 10
    train_episodes = 3
    run_episodes = 3

    # Choose either 'RAM' or 'RGB' for olympia_type
    olympia_type = 'RAM'

    for level in training_levels:
        if olympia_type == 'RAM':
            env = OlympiaRAM(shape=shape, training_level=level)
        elif olympia_type == 'RGB':
            env = OlympiaRGB(shape=shape, training_level=level)
        else:
            raise NotImplementedError('Type not implemented!')
        times, rewards = env.train(episodes=train_episodes, batch_size=batch_size, render=True, load_saved=False, save_models=False)
