from dataclasses import _MISSING_TYPE, dataclass, field

@dataclass
class DefaultConfig:
    save_path:str = 'results'
    pretrained_model: str = ''
    experiments_path:str = 'experiments/experiment.csv'
    seed:int = 0
    model_name: str = 'non_attentive_tacotron'
    process_name: str = 'tgt'
    no_cuda: bool = False
    device: str = 'cuda'

@dataclass
class ForcedAlignerConfig(DefaultConfig):
    model_name: str = 'wav2vec2'
    pretrained_model:str = 'facebook/wav2vec2-base-960h'
    save_grid_path:str = 'textgrid_english'
    save_script_path:str = 'data'
    audio_path: str = ''
    script_path: str = ''
    test_size:int = 10
    normalize_option: str = 'NFKD'


@dataclass
class NonAttentiveTacotronConfig(DefaultConfig):
    pretrained_model: str = ''

    #################### basic training params ###################################
    train_batch_size: int = 32
    eval_batch_size: int = 1
    gradient_accumulation_steps:int = 2
    steps_per_checkpoint:int = 20000
    steps_per_evaluate: int = 2000
    grad_clip_thresh:float = 1.0
    weight_decay:float = 1e-6
    learning_rate:float = 1e-3
    num_train_epochs:int = 500
    max_steps:int  = -1
    warmup_steps: int = -1
    warmup_percent:float = 0.0
    logging_steps:int = 10
    fp16:bool = False
    fp16_opt_level:str="O1"
    n_gpu:int=1
    local_rank:int=-1

    ## generator
    generator_path:str = 'checkpoints_g/vocgan_kss_pretrained_model_epoch_4500.pt'

    ########################## dataset options ###################################
    process_name:str = 'tgt'
    sampling_rate: int = 22050
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    train_script: str = 'data/train_wav2vec3.txt'
    val_script: str = 'data/dev_wav2vec3.txt'
    load_mel_from_disk: bool = False
    normalize_option:str = 'NFKD'

    ############################ model params ######################################
    model_name: str = 'non_attentive_tacotron'

    ## tokenizer
    num_labels: int = 1
    num_special_labels: int = 1

    ## encoder
    symbols_embedding_dim: int = 512
    symbols_special_embedding_dim: int = 32
    encoder_embedding_dim: int = 512
    encoder_dropout_p: float = 0.5
    encoder_kernel_size: int = 5
    encoder_batch_norm_decay: float = 0.999
    encoder_lstm_dim: int = 512 * 2
    encoder_n_convolutions: int = 3
    encoder_activation: str = "relu"

    ## duration predictor
    duration_lstm_dim: int = 512 * 2

    ## range predictor
    range_lstm_dim: int = 512 * 2

    # Positional Embedding
    positional_embedding_dim: int = 32
    positional_timestep: float = 10000.0

    ## decoder
    decoder_lstm_dim: int = 1024
    decoder_lstm_n: int = 2
    decoder_dropout_p: float = 0.1

    ## prenet
    prenet_dim: int = 256
    prenet_activation: str = "relu"
    prenet_dropout_p: float = 0.5

    ## postnet
    postnet_embedding_dim: int = 512
    postnet_activation: str = 'tanh'
    postnet_kernel_size: int = 5
    postnet_n_convolutions: int = 5
    postnet_dropout_p: float = 0.5

    ## loss
    total_duration: bool = False
    in_second: bool = True
    duration_lambda: float = 2.0



