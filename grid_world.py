from pathlib import Path
import pygame
import matplotlib.pyplot as plt

class grid_world:
    def __init__(self):
        pygame.init()
        self.grid_size = 32
        self.xlen = 15
        self.ylen = 15
        self.done=False
        self.clock = pygame.time.Clock()
        self.width = self.grid_size * 15
        self.height = self.grid_size * 15
        self.windows_size = (self.width, self.height)
        self.screen = pygame.display.set_mode(self.windows_size)

        self.agent_img_path = Path("images/agent.png")
        self.reward_path = Path("images/cheese.png")
        self.chip_path = Path("images/chip.png")
        self.img_agent = pygame.transform.smoothscale(
            pygame.image.load(self.agent_img_path), (self.grid_size, self.grid_size)
        )
        self.img_reward = pygame.transform.smoothscale(
            pygame.image.load(self.reward_path), (self.grid_size, self.grid_size)
        )
        self.img_bg = pygame.image.load(self.chip_path)
        self.reward = 0
        self.agent_pos = [1, 1]
        self.reward_pos = [2, 12]
        self.data = self.get_map_data()
        self.cycle = 2
        self.episode_counter = 0
        self.episode_number = 0
        self.reward_sum = 0
        self.running_reward = None
        self.observation = self.agent_pos
        
    def get_map_data(self):
        map_data = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
        1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
        1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]
        return map_data

    def render(self):
        self.chip_list = [
            (0, 0),
            (32, 0),
            (0, 32),
            (32, 32)
        ]
        for y in range(0, 15):
            for x in range(0, 15):
                i = x + y * 15
                c = self.data[i]
                self.screen.blit(self.img_bg,
                        (x*self.grid_size, y*self.grid_size),
                        (self.chip_list[c], (self.grid_size, self.grid_size)))
        self.screen.blit(self.img_agent,
                (self.agent_pos[0]*self.grid_size, self.agent_pos[1]*self.grid_size))
        self.screen.blit(self.img_reward,
                (self.reward_pos[0]*self.grid_size, self.reward_pos[1]*self.grid_size))


    def step(self, action):
        self.action_list = [
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0],
            [0, 0]
        ]
        self.reward_pos_list = [
            [0, 0],
            [2, 2],
            [2, 12],
            [12, 2],
            [12, 12]
        ]
        self.done=False
        self.episode_counter += 1
        self.reward=0.0
        agent_action = self.action_list[action]

        # エージェント位置更新
        self.agent_pos = self.check(agent_action)

        # 報酬・サイクル処理
        if self.agent_pos == self.reward_pos:
            self.cycle = (self.cycle % 4) + 1
            self.reward += 1

        self.reward_pos = self.reward_pos_list[self.cycle]

        # 終了条件
        if self.episode_counter % 1000 == 0:
            self.done = True

        return self.agent_pos, self.reward, self.done


    def check(self, action):
        a_pos = [x+y for (x, y) in zip(self.agent_pos, action)]
        map_data = self.data[a_pos[0] + a_pos[1]*15]
        if map_data == 1:
            return self.agent_pos
        else:
            return a_pos

    def reset(self):
        self.agent_pos = [1, 1]
        self.reward_pos = [2, 12]
        self.cycle = 2
        return self.agent_pos
