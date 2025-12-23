{
    'LR' : 1e-3,
    'EPISODES' : 1000,
    'GAMMA' : 0.99,
    'INPUT_SIZE' : 28,
    'OUTPUT_SIZE' : 4,
    'model_type' : "a2c",
    'use_lstm' : False,

    'env' : 'simple',
    # 'count_vec' : 3, 

    'n_colors' : 25, 
    'size' : (5, 5),
    'pos_goal' : (5, 5),
    'random_pos_agent' : True,
    'seed' : 500,
    'train' : False,
    
    'weights' : './weights/a2c_setup1_simple.pt',
    'seeds_to_eval' : [1356, 2324]
}