import sys

import numpy as np
import pygame
from pygame.locals import *
import time  # 追加
from grid_world import grid_world
from policy_gradient import network
import matplotlib.pyplot as plt

env = grid_world()
observation = env.reset()
xlen = env.xlen
ylen = env.ylen

pg = network(xlen * ylen, 200, 4)
start_time = time.time()
reward_sum = 0
running_reward = None
prev_x = 0
episode_number = 0
reward_history=[]
episode_history=[]
while True:
    #env.render()
    prev_pos = observation
    x = np.zeros(shape=(xlen * ylen), dtype=np.int32)
    x[observation[1] * xlen + observation[0]] = 1
    x[env.reward_pos[1] * xlen + env.reward_pos[0]] = 1

    # 今のネズミの位置
    prev_pos = observation

    # チーズの位置
    prev_reward_pos = env.reward_pos

    aprob = pg.forward(x)
    action = pg.select_action(aprob)
    observation, reward, done = env.step(action)
    reward_pos = env.reward_pos
    agent_pos = observation
    reward_sum += reward
    p_pos = abs(reward_pos[0] - prev_pos[0]) + abs(reward_pos[1] - prev_pos[1])
    c_pos = abs(reward_pos[0] - agent_pos[0]) + abs(reward_pos[1] - agent_pos[1])
    if p_pos > c_pos:
        reward += 1
    else:
        reward -= 1

    pg.record_reward(reward)

    for event in pygame.event.get():
        # 画面の閉じるボタンを押したとき
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    if episode_number == 1000:
        end_time = time.time()
        elapsed_time = end_time - start_time

        print("\n" + "="*30)
        print("学習が完了しました。")
        print(f"総学習時間: {elapsed_time:.2f} 秒")
        print("="*30 + "\n")

        # episode = 1〜1000 を明示
        x = np.arange(1, episode_number + 1)

        plt.figure(figsize=(5, 2.7))

        # 凡例ラベルなし・タイトルなし
        plt.plot(episode_history, reward_history)

        # 軸ラベルは残す（レポートに有用）
        plt.xlabel("episode")
        plt.ylabel("reward")

        # ★ 全条件で比較しやすくするため縦軸固定
        plt.ylim(0, 50)

        # 余白を自動調整
        plt.tight_layout()

        plt.show()

        pygame.quit()
        sys.exit()


    if done:
        episode_number += 1

        if reward_sum != 0:
            pg.backward()

        if episode_number % pg.batch_size == 0:
            pg.update()

        running_reward = (
            reward_sum
            if running_reward is None
            else running_reward * 0.99 + reward_sum * 0.01
        )
        print(
            "ep %d:  episode reward total was %f. running mean: %f"
            % (episode_number, reward_sum, running_reward)
        )
        episode_history.append(episode_number)
        reward_history.append(reward_sum)
        reward_sum = 0
        observation = env.reset()  # reset env
        
    #pygame.display.flip()
