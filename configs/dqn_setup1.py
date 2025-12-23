{
    'BATCH_SIZE' : 128,
    'LR' : 1e-3,
    'MEMORY_SIZE' : 50_000,
    'EPISODES' : 1000,
    'EPSILON' : 0.3,
    'GAMMA' : 0.99,
    'TARGET_UPDATE_FREQ' : 10,
    'INPUT_SIZE' : 28,
    'OUTPUT_SIZE' : 4,
    "model_type" : "dqn",
    'use_lstm' : False,

    'env' : 'simple',
    # 'count_vec' : 3, 

    'n_colors' : 25, 
    'size' : (5, 5),
    'pos_goal' : (5, 5),
    'random_pos_agent' : True,
    'seed' : 500,
    'train' : False,

    'weights' : './weights/dqn_setup1_simple.pt',
    'seeds_to_eval' : [1356, 2324]
}