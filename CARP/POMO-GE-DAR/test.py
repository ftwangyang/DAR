##########################################################################################
# Machine Environment Config

import time

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

from CARPTester import CARPTester as Tester


##########################################################################################
# parameters

env_params = {
    'vertex_size': 50,
    'edge_size': 100,
    'pomo_size': 100,
}

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
    'omega': 0.3,
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/carp_v50e100_DAR_test_k=101_w=0.3',  # directory path of pre-trained model and log files saved.
        # 'path': './result/carp_v50e100_DAR_test',
        'epoch': 300,  # epoch version of pre-trained model to laod.
    },
    'test_data_load': {
        'enable': True,
        'filename': '../CARP/node50edge100_num=100'
    },
    'test_episodes': 100,
    'test_batch_size': 100,
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    _print_config()

    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)

    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)

    tester.run()
    # time1 = time.time()
    # for i in range(10):
    #     tester.run()
    # Totaltime = time.time() - time1
    # print("Total time: {}".format(Totaltime))
    # print("Average time: {}".format(Totaltime/10))



def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":

    main()

