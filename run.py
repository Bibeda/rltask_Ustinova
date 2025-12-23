import os
import sys
import time
from environments import GridWorldEnv, MNISTWorldEnv
from agents.a2c import A2C, RecurrentA2C, A2CSingleTrainer, A2CVectorTrainer
from agents.dqn import DQN, RecurrentDQN, DQNSingleTrainer, DQNVectorTrainer
from agents.dqn import ReplayMemory, BufferMemory
from gymnasium.vector import SyncVectorEnv


def make_env(h, w, pos_goal, **env_kwargs):
    def _init():
        env = GridWorldEnv(
            h=h,
            w=w,
            pos_goal=pos_goal,
            **env_kwargs
        )
        return env
    return _init

def make_vec(num_envs, h, w, pos_goal, env_kwargs=None, **vector_kwargs):
    if env_kwargs is None:
        env_kwargs = {}

    env_fns = [
        make_env(h, w, pos_goal, **env_kwargs) 
        for _ in range(num_envs)
    ]

    env = SyncVectorEnv(
        env_fns=env_fns,
        **vector_kwargs
    )
    return env


def create_enviroment(config):
    h, w = config['size']
    if config['env'] == 'simple':
        env = GridWorldEnv(h, w, pos_goal=config['pos_goal'], n_colors=config['n_colors'], random_pos_agent=config['random_pos_agent'], seed=config['seed'])
    elif config['env'] == 'vector':
        env = make_vec(config['count_vec'], h, w, pos_goal=config['pos_goal'], env_kwargs={'random_pos_agent':config['random_pos_agent'], 'seed':config['seed'], 'n_colors':config['n_colors']})

    return env

def create_model(config, env):
    if config['model_type'] == 'a2c':
        if config['use_lstm']:
            model = RecurrentA2C(config['INPUT_SIZE'], config['OUTPUT_SIZE'], config['LSTM_HIDDEN_DIM'])
        else:
            model = A2C(config['INPUT_SIZE'], config['OUTPUT_SIZE'], 128)
        
        trainer = A2CSingleTrainer(env, config=config, model=model) if config['env'] == 'simple' else A2CVectorTrainer(env, config=config, model=model)
    else:
        if config['use_lstm']:
            policy_model = RecurrentDQN(config['INPUT_SIZE'], config['OUTPUT_SIZE'], config['LSTM_HIDDEN_DIM'])
            target_model = RecurrentDQN(config['INPUT_SIZE'], config['OUTPUT_SIZE'], config['LSTM_HIDDEN_DIM'])
            memory = BufferMemory(config['MEMORY_SIZE'], config['TIME_STEP'])
        else:
            policy_model = DQN(config['INPUT_SIZE'], config['OUTPUT_SIZE'], hidden_dim=32)
            target_model = DQN(config['INPUT_SIZE'], config['OUTPUT_SIZE'], hidden_dim=32)
            memory = ReplayMemory(config['MEMORY_SIZE'])
        
        trainer = DQNSingleTrainer(env, config, policy_model, target_model, memory) if config['env'] == 'simple' else DQNVectorTrainer(env, config, policy_model, target_model, memory)
    return trainer


def run(config):
    env = create_enviroment(config)

    trainer = create_model(config, env)
    
    if config['train']:
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        print(f"Обучение заняло: {training_time/60:.2f} минут")
        trainer.save(config['weights'])
        trainer.plot_loss()
        trainer.plot_window_reward()
        if config['model_type'] == 'a2c':
            trainer.plot_a2c_losses()
    else:
        trainer.load(config['weights'])
    for i in config['seeds_to_eval']:
        trainer.evaluate(i)
    
    return



def main():
    if len(sys.argv) < 2:
        print("Укажите путь к файлу как аргумент")
        return
    
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = eval(f.read())
    config['name'] = config['env'] + "_" + os.path.basename(config_path).split('.')[0]

    run(config)
    return

if __name__ == "__main__":
    main()