from olympia.envs import OlympiaRGB, OlympiaRAM

if __name__ == "__main__":
    shape = (15, 10)
    training_level = 'one player'
    batch_size = 25
    train_episodes = 3
    run_episodes = 3
    # Choose either 'RAM' or 'RGB' for olympia_type
    olympia_type = 'RAM' 

    if olympia_type == 'RAM':
        env = OlympiaRAM(shape=shape, training_level=training_level)
    elif olympia_type == 'RGB':
        env = OlympiaRGB(shape=shape, training_level=training_level)
    else:
        raise NotImplementedError('Type not implemented!')
    env.train(episodes=train_episodes, batch_size=batch_size, render=False, load_saved=False)
    env.run(episodes=run_episodes,render=True)