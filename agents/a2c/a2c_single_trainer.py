from ..base_trainer import BaseTrainer
import torch
import numpy as np
import time

class A2CSingleTrainer(BaseTrainer):
    def __init__(self, env, config, model=None, buffer=None):
        super().__init__(env, model, config)
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['LR'])
        self.buffer = buffer
        self.episode_times = []

    def train(self):
        for episode in range(self.config['EPISODES']):
            episode_start = time.time()
            log_probs, rewards, values = self._collect_data()
            returns = []
            discounted_return = 0
            for r in reversed(rewards):
                discounted_return = r + self.config['GAMMA'] * discounted_return
                returns.insert(0, discounted_return)

            returns = torch.FloatTensor(returns)
            log_probs = torch.stack(log_probs)
            values = torch.cat(values).squeeze()

            advantages = returns - values

            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()
            loss = actor_loss + critic_loss

            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(critic_loss.item())
            self.avg_losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            episode_duration = time.time() - episode_start
            self.episode_times.append(episode_duration)
            has_rewards = sum(rewards)
            self.all__rewards.append(has_rewards)
            if episode % 100 == 0:
                print(f"Episode {episode}: Reward = {has_rewards}, Current Loss = {self.avg_losses[-1]}")
                print(f"  Actor Loss: {self.actor_losses[-1]:.4f}, Critic Loss: {self.critic_losses[-1]:.4f}")
        print()
        print(f"Count Episode: {self.config['EPISODES']} | Mean time of episode: {np.mean(self.episode_times)}")
        print()

    def _collect_data(self):
        state, _ = self.env.reset()
        state = self._prepare_state(state)
        log_probs, rewards, values = [], [], []
        done = False
        model_state = {}

        if self.config['use_lstm']:
            model_state['hidden'] = self.model.init_hidden_states(bsize=1)

        while not done:
            if self.config['use_lstm']:
                action_probs, value, model_state['hidden'] = self.model(state, model_state['hidden'])
            else:
                action_probs, value = self.model(state)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs.squeeze(0)[action])
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = self._prepare_state(next_state)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

            state = next_state

            done = (terminated | truncated)

            if done:
                break

        return log_probs, rewards, values


    def _prepare_state(self, state, is_evaluate=False):
        state = torch.from_numpy(state).float()
        if self.config['use_lstm']:
            return state.view(1, 1, -1)
        return state.unsqueeze(0)
    