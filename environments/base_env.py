import gymnasium as gym
import numpy as np


class GridWorldEnv(gym.Env):

    def __init__(self,
                 h,
                 w,
                 pos_goal,
                 n_colors=None,
                 pos_agent=None,
                 random_pos_agent=True,
                 see_obstacle=True,
                 obstacle_mask=None,
                 random_obstacle=False,
                 obstacle_prob=0.1,
                 steps=100, 
                 seed=None):
        super().__init__()
        self.h = h
        self.w = w
        self.n_colors = (h * w) if n_colors is None else n_colors
        self.pos_goal = pos_goal
        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,
                                                shape=(self.n_colors + 3, ))
        self.action_space = gym.spaces.Discrete(4)
        #вверх, вправо, вниз, влево
        self.directions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.max_episode_steps = steps  # ???
        if (pos_agent is None) ^ random_pos_agent:
            raise ValueError(
                "Нужен или только pos_agent или random_pos_agent=True")
        if obstacle_mask is not None and random_obstacle:
            raise ValueError(
                "Задайте только obstacle_mask ИЛИ random_obstacle=True")
        
        if seed is not None:
            np.random.seed(seed)

        self.obstacle_mask = self.__init_obstacle(
            pos_agent, obstacle_prob) if random_obstacle else obstacle_mask
        self.see_obstacle = see_obstacle
        self.pos_agent = self.__init_pos_agent(
        ) if random_pos_agent else pos_agent
        self.cur_pos_agent = self.__init_cur_pos_agent()
        self.grid = self.__init_grid()

    def __init_obstacle(self, pos_agent, obstacle_prob):
        all_positions = [(y, x) for y in range(self.h) for x in range(self.w)]
        all_positions.remove((self.pos_goal[0] - 1, self.pos_goal[1] - 1))
        if pos_agent is not None:
            all_positions.remove(
                (self.pos_agent[0] - 1, self.pos_agent[1] - 1))
        cnt_obstacles = int(round(self.h * self.w * obstacle_prob))
        chosen = np.random.choice(len(all_positions), size=cnt_obstacles, replace=False)
        obstacle_positions = [all_positions[i] for i in chosen]
        obstacle_mask = np.zeros(shape=(self.h, self.w))
        for y, x in obstacle_positions:
            obstacle_mask[y, x] = 1
        return obstacle_mask

    def __init_pos_agent(self):
        random_arr = np.full((self.h + 2, self.w + 2), 0, dtype=int)
        random_arr[1:-1, 1:-1] = np.random.randint(1, 5, size=(self.h, self.w))
        if self.obstacle_mask is not None:
            obstacle_r, obstacle_c = np.where(self.obstacle_mask == 1)
            random_arr[obstacle_r + 1, obstacle_c + 1] = 0
        random_arr[self.pos_goal[0], self.pos_goal[1]] = 0
        return random_arr

    def __init_cur_pos_agent(self):
        if isinstance(self.pos_agent, np.ndarray):
            prob_arr = (self.pos_agent / np.sum(self.pos_agent)).flatten()
            idx = np.random.choice(len(prob_arr), p=prob_arr)
            return (idx // (self.w + 2), idx % (self.w + 2))
        return self.pos_agent

    def __init_grid(self):
        grid = np.full((self.h + 2, self.w + 2), -1, dtype=int)
        if self.n_colors == self.h * self.w:
            grid[1:-1, 1:-1] = np.random.permutation(self.h * self.w).reshape(
                self.h, self.w)
        else:
            grid[1:-1, 1:-1] = np.random.randint(0,
                                                 self.n_colors,
                                                 size=(self.h, self.w))
        if self.obstacle_mask is not None:
            obstacle_r, obstacle_c = np.where(self.obstacle_mask == 1)
            grid[obstacle_r + 1, obstacle_c + 1] = -2
        return grid

    def _get_observation(self, pos):
        # [n_colors, goal, wall, obstacle]
        obs = np.zeros(self.n_colors + 3)
        r, c = pos
        cell_value = self.grid[r, c]
        if cell_value == -1:
            obs[-2] = 1
        elif cell_value == -2:
            obs[-1] = 1
        elif pos == self.pos_goal:
            obs[-3] = 1
        else:
            obs[cell_value] = 1
        return obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.cur_pos_agent = self.__init_cur_pos_agent()
        self.step_count = 0
        return self._get_observation(self.cur_pos_agent), {}  # obs, info

    def step(self, action):
        self.step_count += 1
        dr, dc = self.directions[action]
        cur_r, cur_c = self.cur_pos_agent
        r, c = cur_r + dr, cur_c + dc
        new_pos = (r, c)
        obs, reward, terminated, truncated = None, 0, False, (
            self.step_count >= self.max_episode_steps)
        if self.grid[r, c] < 0:
            if self.see_obstacle:
                obs = self._get_observation(new_pos)
            else:
                obs = self._get_observation(self.cur_pos_agent)
        else:
            obs = self._get_observation(new_pos)
            self.cur_pos_agent = new_pos
            if new_pos == self.pos_goal:
                reward = 1
                terminated = True
        return obs, reward, terminated, truncated, {}

    def print_grid_terminal(self):
        grid = self.grid.copy().astype(str)
        r, c = self.cur_pos_agent
        rr, cc = self.pos_goal
        grid[r, c] = "*"
        grid[rr, cc] = "@"
        grid[grid == "-1"] = "#"
        grid[grid == "-2"] = "#"
        return grid