from olympia.envs import OlympiaRGB, OlympiaRAM

if __name__ == "__main__":
    shape = (21, 15)
    training_level = 'one v one'
    batch_size = 25
    train_episodes = 1000
    run_episodes = 3

    env = OlympiaRAM(shape=shape, training_level=training_level)
    env.train(episodes=train_episodes, batch_size=batch_size, render=True, load_saved=False)
    env.run(episodes=run_episodes,render=True)