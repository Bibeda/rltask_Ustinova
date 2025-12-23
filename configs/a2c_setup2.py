{
    'LR' : 1e-3,
    'EPISODES' : 2000,
    'GAMMA' : 0.99,
    'INPUT_SIZE' : 8,
    'OUTPUT_SIZE' : 4,
    'LSTM_HIDDEN_DIM' : 128, 
    'model_type' : "a2c",
    'use_lstm' : True,
    'env' : 'simple',

    'n_colors' : 5, 
    'size' : (5, 5),
    'pos_goal' : (5, 5),
    'random_pos_agent' : True,
    'seed' : 500,
    'train' : False,

    'weights' : './weights/a2c_setup2_simple.pt',
    'seeds_to_eval' : [622, 98]
}
