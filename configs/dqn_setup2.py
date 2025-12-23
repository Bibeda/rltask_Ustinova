{
    'BATCH_SIZE' : 64,
    'LR' : 1e-3,
    'MEMORY_SIZE' : 50_000,
    'EPISODES' : 2000,
    'EPSILON' : 0.2,
    'GAMMA' : 0.99,
    'TARGET_UPDATE_FREQ' : 10,
    'TIME_STEP' : 8,
    'INPUT_SIZE' : 8,
    'OUTPUT_SIZE' : 4,
    'LSTM_HIDDEN_DIM' : 128, 
    'use_lstm' : True,
    'model_type' : "dqn",

    'env' : "simple",
    'count_vec' : 3,

    'n_colors' : 5, 
    'size' : (5, 5),
    'pos_goal' : (5, 5),
    'random_pos_agent' : True,
    'seed' : 500,
    'train' : False,

    'weights' : './weights/dqn_setup2_simple.pt',
    'seeds_to_eval' : [622, 98]
}