from ..base_trainer import BaseTrainer
import torch
import numpy as np
import torch.nn.functional as F


class DQNVectorTrainer(BaseTrainer):
    def __init__(self, env, config, model=None, target_model=None, buffer=None):
        super().__init__(env, model, config)
        self.model = model 
        self.target_model = target_model
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['LR'], amsgrad=True)
        self.buffer = buffer
        self.num_envs = env.num_envs if hasattr(self.env, 'num_envs') else 1


    def train(self):
        for episode in range(self.config['EPISODES']):
            state, _ = self.env.reset()
            state = self._prepare_state(state) #for lstm shape = [batch=1, seq_len=1, features] for mlp shape = [batch=1, features]
            all_dones, cnt_rewards = np.zeros(self.env.num_envs, dtype=bool), np.zeros(self.env.num_envs, dtype=float)
            model_state = {}
            current_episode_loss, num_upd = 0, 0

            if self.config['use_lstm']:
                model_state['hidden'] = self.model.init_hidden_states(bsize=self.num_envs)
                self.local_memory_for_lstm = []

            while not any(all_dones):
                action, model_state = self._select_action(state, model_state)
                next_state, reward, terminated, truncated, _ = self.step(action, np.where(~all_dones)[0])
                next_state = self._prepare_state(next_state)
                self._save_data(state, action, next_state, reward, terminated, truncated)
                if len(self.buffer) > self.config['BATCH_SIZE']:
                    num_upd += 1
                    current_episode_loss += self._update_model(model_state)
                cnt_rewards += reward
                state = next_state
                all_dones = (terminated | truncated)

            if self.config['use_lstm']:
                self.buffer.push(self.local_memory_for_lstm)

            if episode % self.config['TARGET_UPDATE_FREQ'] == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            if num_upd > 0:
                self.avg_losses.append(current_episode_loss/num_upd)

            if episode % 10 == 0:
                if num_upd > 0:
                    print(f"Episode: {episode}, Num_updates: {num_upd}, Avg_loss: {self.avg_losses[-1]:.6f}, Get_rewards: {cnt_rewards}")
                cnt_rewards = np.zeros(self.env.num_envs, dtype=float)

    def _select_action(self, state, model_state):
        action = np.zeros(self.num_envs, dtype=int)
        is_random_action = np.random.rand(self.num_envs) < self.config['EPSILON']
        random_action = np.random.randint(low=0, high=self.config['OUTPUT_SIZE'], size=self.num_envs)
        with torch.no_grad():
            if self.config['use_lstm']:
                q_vals, model_state['hidden'] = self.model(state, model_state['hidden'])
            else:
                q_vals = self.model(state)
        q_vals = torch.argmax(q_vals, dim=-1)
        action = np.where(is_random_action == 1, q_vals, random_action)
        return action, model_state


    def _update_model(self, model_state):
        if self.config['use_lstm']:
            model_state['hidden_batch'] = self.model.init_hidden_states(bsize=self.config['BATCH_SIZE'])
            states, actions, next_states, rewards, terminated_batch, truncated_batch = self.buffer.get_batch(self.config['BATCH_SIZE'])
            Q_values, _ = self.model(states, model_state['hidden_batch']) # [batch_size, time_step, 4]
            Q_values_action = Q_values.gather(1, actions[:, -1].unsqueeze(1)).squeeze(1) # q-value выбранных действий
            with torch.no_grad():
                Q_next, _ = self.target_model(next_states, model_state['hidden_batch'])
                Q_next_max = Q_next.max(1).values # выбираем максимальное q-value у последних действий
                dones = (terminated_batch.bool() | truncated_batch.bool()).float()
                dones_last = dones[:, -1] 
                target_Q_values = rewards[:, -1] + self.config['GAMMA'] * Q_next_max * (1 - dones_last)
        else:
            states, actions, next_states, rewards, terminated_batch, truncated_batch = self.buffer.sample(self.config['BATCH_SIZE'])
            Q_values = self.model(states)
            Q_values_action = Q_values.gather(1, actions).squeeze(1)
            with torch.no_grad():
                Q_next = self.target_model(next_states)
                Q_next_max = Q_next.max(1).values
                dones = (terminated_batch.bool() | truncated_batch.bool()).float()
                target_Q_values = rewards + self.config['GAMMA'] * Q_next_max * (1 - dones)

        loss = F.mse_loss(Q_values_action, target_Q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _prepare_state(self, state, is_evaluate=False):
        state = torch.from_numpy(state).float()
        if self.num_envs > 1 and not is_evaluate:
            if self.config['use_lstm']:
                state = state.unsqueeze(1)
                return state
            else:
                return state

        if self.config['use_lstm']:
            return state.view(1, 1, -1) # [batch, seq_len, features]
        return state.unsqueeze(0) # makr [1, features]

    def _save_data(self, state, action, next_state, reward, terminated, truncated):
        if not self.config['use_lstm']:
            for i in range(state.shape[0]):
                self.buffer.push(state[i].unsqueeze(0), action[i], next_state[i].unsqueeze(0), reward[i], terminated[i], truncated[i])
        else:
            for i in range(state.shape[0]):
                self.local_memory_for_lstm.append((state[i].flatten(), action[i], next_state[i].flatten(), reward[i], terminated[i], truncated[i]))
        return
    
    def step(self, actions, alive_index):
        act = np.zeros(self.env.num_envs, dtype=int)
        for i, j in zip(alive_index, actions):
            act[i] = j
        obs, reward, terminated, truncated, _ = self.env.step(act)
        obs_alive, reward_alive, terminated_alive, truncated_alive = obs[alive_index], reward[alive_index], terminated[alive_index], truncated[alive_index]
        return obs_alive, reward_alive, terminated_alive, truncated_alive, _
