##########################################################################################
# Machine Environment Config
import random

import numpy as np
import torch

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from CARPTrainer import CARPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    'vertex_size': 50,
    'edge_size': 100,
    'pomo_size': 100,
}
task = 'v'+str(env_params['vertex_size'])+'e'+str(env_params['edge_size'])

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',

    'mask_num': 101,
    'omega': 0.9,

}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [301, 351],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 300,
    'train_episodes': 128 * 1000,
    'train_batch_size': 128,
    'prev_model_path': None,
    'logging': {
        'model_save_interval': 10,
        # 'img_save_interval': 500,
        # 'log_image_params_1': {
        #     'json_foldername': 'log_image_style',
        #     'filename': 'style_cvrp_100.json'
        # },
        # 'log_image_params_2': {
        #     'json_foldername': 'log_image_style',
        #     'filename': 'style_loss_1.json'
        # },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './result/carp_v10e20_GCNnolinear',  # directory path of pre-trained model and log files saved.
        'epoch': 300,  # epoch version of pre-trained model to laod.

    }
}

logger_params = {
    'log_file': {
        'desc': 'carp_'+task+"_DAR_test_k=101_w=0.9",
        'filename': 'run_log'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    seed = 1234  # 可以是任何你选择的整数种子
    # 设置 Python 随机种子
    random.seed(seed)
    # 设置 NumPy 随机种子
    np.random.seed(seed)
    # 设置 PyTorch 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(
                      env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params = trainer_params)


    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
