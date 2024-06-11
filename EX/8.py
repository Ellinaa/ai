import gym

env = gym.make("CartPole-v1")

# 開始遊戲迴圈
total_steps = 0  # 總步數，即撐的時間

for _ in range(100):
    observation = env.reset()
    done = False

    while not done:
        env.render()
        angle = observation[2]

        if angle < 0:
            action = 0  # 向左移動
        else:
            action = 1  # 向右移動

        observation, reward, done, info = env.step(action)
        total_steps += 1  # 每迴圈增加一步

        if done:
            print('done')
            print('Total Steps:', total_steps)
            total_steps = 0  # 重置總步數

env.close()
