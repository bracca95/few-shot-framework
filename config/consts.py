# about TypedDict https://stackoverflow.com/a/64938100


class ConfigConst:
    CONFIG_EXPERIMENT_NAME = "experiment_name"
    CONFIG_DATASET = "dataset"
    CONFIG_MODEL = "model"
    CONFIG_TRAIN_TEST = "train_test"

class ModelConfig:
    CONFIG_MODEL_NAME = "model_name"
    CONFIG_FREEZE = "freeze"
    CONFIG_FSL = "fsl"

class FSLConsts:
    FSL_EPISODES = "episodes"
    FSL_TRAIN_N_WAY = "train_n_way"
    FSL_TRAIN_K_SHOT_S = "train_k_shot_s"
    FSL_TRAIN_K_SHOT_Q = "train_k_shot_q"
    FSL_TEST_N_WAY = "test_n_way"
    FSL_TEST_K_SHOT_S = "test_k_shot_s"
    FSL_TEST_K_SHOT_Q = "test_k_shot_q"
    FSL_ENHANCEMENT = "enhancement"

class TrainTestConfig:
    CONFIG_EPOCHS = "epochs"
    CONFIG_BATCH_SIZE = "batch_size"
    CONFIG_MODEL_TEST_PATH = "model_test_path"
    CONFIG_LEARNING_RATE = "learning_rate"
    CONFIG_OPTIMIZER = "optimizer"
