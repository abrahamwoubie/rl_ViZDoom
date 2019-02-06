class GlobalVariables :

    #options for running different experiments
    use_MFCC = 0
    use_Pixels=1
    use_samples=0

    #parameters
    how_many_times_training=1
    how_many_times = 200000
    replay_memory_size = 100000
    replay_memory_batch_size = 64

    Learning_Rate = 0.00025
    Discount_Factor = 0.99

    frame_repeat = 10
    channels = 1

    channels_audio = 1

    start_eps = 1.0
    end_eps = 0.1
    eps_decay_iter = 0.33 * how_many_times

    save_each = 0.01 * how_many_times
