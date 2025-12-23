import torch
import matplotlib.pyplot as plt
import numpy as np

class BaseTrainer:
    def __init__(self, env, model, config):
        self.env = env
        self.config = config
        self.model = model
        self.avg_losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.all__rewards = []

    def train(self):
        raise NotImplementedError
    
    def _get_env(self):
        if hasattr(self.env, 'num_envs') and self.env.num_envs > 1:
            return self.env.envs[0]
        return self.env
    
    def evaluate(self, seed):
        to_output = ""
        env = self._get_env()
        if seed is not None:
            state, _ = env.reset(seed)
        else:
            state, _ = env.reset()
        to_output += str(env.print_grid_terminal()) + "\n"
        state = self._prepare_state(state, is_evaluate=True)
        self.trajectory = []
        done = False
        if self.config['use_lstm']:
            hidden = self.model.init_hidden_states(bsize=1)
        env.print_grid_terminal()

        while not done:
            last_pos = env.cur_pos_agent
            with torch.no_grad():
                if self.config['model_type'] == 'a2c':
                    if self.config['use_lstm']:
                        action_probs, _, hidden = self.model(state, hidden)
                    else:
                        action_probs, _ = self.model(state)
                else:
                    if self.config['use_lstm']:
                        action_probs, hidden = self.model(state, hidden)
                    else:
                        action_probs = self.model(state)
                action = torch.argmax(action_probs).item()
            self.trajectory.append((last_pos, action_probs, action))
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = self._prepare_state(next_state, is_evaluate=True)

            d = {
            0: "up",
            1: "right",
            2: "down",
            3: "left"
            }

            to_output += str(action_probs) + " --> " + str(d[action]) + "\n"
            to_output += str(env.print_grid_terminal())

            state = next_state
            done = (terminated | truncated)
        self.trajectory.append((env.cur_pos_agent, [], None))
        
        with open(self.config['name'], 'w') as f:
            f.write(to_output)
        
        self.render_trajectory()

    def plot_loss(self, group=10):
        avg_loss_per_group = [np.mean(self.avg_losses[i:i+group]) for i in range(0, len(self.avg_losses), group)]
        plt.plot(avg_loss_per_group)
        plt.xlabel(f'Groups of {group} updates')
        plt.ylabel('Loss')
        plt.show()

    def plot_a2c_losses(self, group=10):
        if self.config['model_type'] != 'a2c':
            raise TypeError("This method is only for A2C models")
        avg_actor = [np.mean(self.actor_losses[i:i+group]) for i in range(0, len(self.actor_losses), group)]        
        avg_critic = [np.mean(self.critic_losses[i:i+group]) for i in range(0, len(self.critic_losses), group)]        

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        #actor loss
        ax1.plot(avg_actor, color='blue', linewidth=2)
        ax1.set_title('Actor Loss (Policy)')
        ax1.set_xlabel(f'Groups of {group} updates')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        #critic loss
        ax2.plot(avg_critic, color='red', linewidth=2)
        ax2.set_title('Critic Loss (Value)')
        ax2.set_xlabel(f'Groups of {group} updates')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_window_reward(self, window=20):
        # print(self.all__rewards)
        avg_reward = [np.mean(self.all__rewards[i:i+window]) * 100 for i in range(0, len(self.all__rewards), window)]        
        plt.plot(avg_reward)
        plt.xlabel(f'Эпизод (скользящее окно={window})')
        plt.ylabel('Успех (%)')
        plt.title('Процент успешных эпизодов')
        plt.show()


    def render_trajectory(self):
        env = self._get_env()
        grid = env.grid
        cmap = plt.cm.viridis  
        mask = grid >= 0
        colored = np.zeros((*grid.shape, 3))
        colored[mask] = [1, 1, 1]  # поле
        colored[~mask] = [0, 0, 0]  # граница и препятствия

        visit_counts = np.zeros(grid.shape)
        for pos, _, _ in self.trajectory:
            visit_counts[pos] += 1

        if visit_counts.max() > 0:
            visit_norm = visit_counts / visit_counts.max()
        else:
            visit_norm = visit_counts

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if visit_counts[i, j] > 0:
                    colored[i, j] = cmap(visit_norm[i, j])[:3]
        
        start = self.trajectory[0][0]
        goal = env.pos_goal
        
        colored[start] = [0, 1, 0]
        colored[goal] = [1, 0, 0]
        plt.imshow(colored)
        plt.axis('off')
        plt.show()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))