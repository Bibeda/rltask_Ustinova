from ..base_trainer import BaseTrainer
import torch
import numpy as np

class A2CVectorTrainer(BaseTrainer):
    def __init__(self, env, config, model=None):
        super().__init__(env, model, config)
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['LR'])
        self.num_envs = env.num_envs if hasattr(self.env, 'num_envs') else 1
        
    def train(self):
        print(f"Train Vectorize A2C. Envs = {self.num_envs}")
        print()
        for episode in range(self.config['EPISODES']): 
            log_probs, rewards, values, terminated_stat, steps_stat = self._collect_data()
            all_log_probs, all_returns, all_values = [], [], []
            
            for i in range(self.num_envs):
                discounted_return = 0
                ret = []
                for r in reversed(rewards[i]):
                    discounted_return = r + self.config['GAMMA'] * discounted_return
                    ret.insert(0, discounted_return)
                all_returns.extend(ret)
                all_log_probs.extend(log_probs[i])
                all_values.extend(values[i])

            returns = torch.FloatTensor(all_returns)
            log_probs = torch.stack(all_log_probs)
            values = torch.cat(all_values).squeeze()

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

            if episode % 100 == 0:
                print(f"Episode {episode}: Reward on {sum(terminated_stat)} plays, Step count {steps_stat}")
                print(f"    Current Loss = {self.avg_losses[-1]:.6f}")
                print(f"    Actor Loss: {self.actor_losses[-1]:.6f}, Critic Loss: {self.critic_losses[-1]:.6f}")
                print()


    def _collect_data(self):
        state, _ = self.env.reset()
        state = self._prepare_state(state)
        all_log_probs, all_rewards, all_values = [[] for _ in range(self.num_envs)], [[] for _ in range(self.num_envs)], [[] for _ in range(self.num_envs)]
        all_dones, terminated_stat = np.zeros(self.env.num_envs, dtype=bool), np.zeros(self.env.num_envs, dtype=bool)
        steps_stat = np.ones(self.env.num_envs, dtype=int)
        alive_idx = np.where(~all_dones)[0]
        model_state = {}

        if self.config['use_lstm']:
            model_state['hidden'] = self.model.init_hidden_states(bsize=self.num_envs)

        while not all(all_dones):
            if self.config['use_lstm']:
                action_probs, value, model_state['hidden'] = self.model(state, model_state['hidden'])
            else:
                action_probs, value = self.model(state)
            actions = torch.multinomial(action_probs, 1)
            log_prob = torch.log(action_probs.gather(1, actions).squeeze(1))
            actions_np = actions.cpu().numpy().flatten()
            next_state, reward, terminated, truncated, _ = self.env.step(actions_np)
            all_dones |= (terminated | truncated)
            terminated_stat |= terminated

            for j, env_idx in enumerate(alive_idx):
                all_log_probs[env_idx].append(log_prob[j])
                all_rewards[env_idx].append(reward[j])
                all_values[env_idx].append(value[j])  
                steps_stat[env_idx] += 1

            alive_idx = np.where(~all_dones)[0]
            state = self._prepare_state(next_state)

        return all_log_probs, all_rewards, all_values, terminated_stat, steps_stat


    def _prepare_state(self, state, is_evaluate=False):
        state = torch.from_numpy(state).float()
        if self.num_envs > 1 and not is_evaluate:
            if self.config['use_lstm']:
                state = state.unsqueeze(1)
                return state
            else:
                return state
            
        if self.config['use_lstm']:
            return state.view(1, 1, -1)
        return state.unsqueeze(0)
