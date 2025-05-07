

SAVE_FOLDER_NAME = "INFERENCE_00"
print(SAVE_FOLDER_NAME)



from HYPER_PARAMS import *  # NOTE : You much edit HYPER_PARAMS to match the model you are loading
from TORCH_OBJECTS import *
import numpy as np
import time


# In[ ]:


from source.knapsack_problem import GROUP_ENVIRONMENT
import source.MODEL__Actor.grouped_actors as A_Module
from source.utilities import Average_Meter, Get_Logger

import random
import numpy as np
import torch

SEED = 2026  # 你可以选择任何整数作为随机种子

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Make Log File
logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)

print(result_folder_path)


# Load Model

grouped_actor = A_Module.ACTOR().to(device)

actor_model_save_path = './result/Saved_KP200_Model/ACTOR_state_dic.pt'
grouped_actor.load_state_dict(torch.load(actor_model_save_path, map_location="cuda:0"))
grouped_actor.eval()

logger.info('==============================================================================')
logger.info('==============================================================================')
log_str = '  <<< MODEL: {:s} >>>'.format(actor_model_save_path)
logger.info(log_str)



LOG_PERIOD_SEC = 10

DATASET_SIZE = 100 *1
TEST_BATCH_SIZE = 10



eval_dist_AM_0 = Average_Meter()


logger.info('===================================================================')
log_str = 'Single Trajectory'
logger.info(log_str)

timer_start = time.time()
logger_start = time.time()


episode = 0
while True:

    data = Tensor(np.random.rand(TEST_BATCH_SIZE, PROBLEM_SIZE, 2))

    batch_s = data.size(0)
    episode = episode + batch_s


    with torch.no_grad():
 
        env = GROUP_ENVIRONMENT(data)
        group_s = 1
        group_state, reward, done = env.reset(group_size=group_s)
        grouped_actor.reset(group_state)

        # First Move is given
        first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
        group_state, reward, done = env.step(first_action)

        while not done:
            action_probs = grouped_actor.get_action_probabilities(group_state)
            # shape = (batch, group, problem)
            action = action_probs.argmax(dim=2)
            # shape = (batch, group)

            action_w_finished = action.clone()
            action_w_finished[group_state.finished] = PROBLEM_SIZE  # this is dummy item with 0 size 0 value
            group_state, reward, done = env.step(action_w_finished)

    eval_dist_AM_0.push(reward)  



    if (time.time()-logger_start > LOG_PERIOD_SEC) or (episode >= DATASET_SIZE):
        timestr = time.strftime("%H:%M:%S", time.gmtime(time.time()-timer_start))
        log_str = 'Ep:{:07d}({:5.1f}%)  T:{:s}  avg:{:f}'            .format(episode, episode/DATASET_SIZE*100,
                    timestr, eval_dist_AM_0.peek())
        logger.info(log_str)
        logger_start = time.time()
        
    if episode >= DATASET_SIZE:
        break

    
logger.info('---------------------------------------------------')
logger.info('average = {}'.format(eval_dist_AM_0.result()))
logger.info('---------------------------------------------------')
logger.info('---------------------------------------------------')



eval_dist_AM_0 = Average_Meter()

logger.info('===================================================================')
log_str = 'All Trajectory'
logger.info(log_str)


timer_start = time.time()
logger_start = time.time()
episode = 0
while True:

    data = Tensor(np.random.rand(TEST_BATCH_SIZE, PROBLEM_SIZE, 2))

    batch_s = data.size(0)
    episode = episode + batch_s


    with torch.no_grad():

        env = GROUP_ENVIRONMENT(data)
        group_s = PROBLEM_SIZE
        group_state, reward, done = env.reset(group_size=group_s)
        grouped_actor.reset(group_state)

        # First Move is given
        first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
        group_state, reward, done = env.step(first_action)

        while not done:
            action_probs = grouped_actor.get_action_probabilities(group_state)
            # shape = (batch, group, problem)
            action = action_probs.argmax(dim=2)
            # shape = (batch, group)

            action_w_finished = action.clone()
            action_w_finished[group_state.finished] = PROBLEM_SIZE  # this is dummy item with 0 size 0 value
            group_state, reward, done = env.step(action_w_finished)


    max_reward, _ = reward.max(dim=1)
    eval_dist_AM_0.push(max_reward)  



    if (time.time()-logger_start > LOG_PERIOD_SEC) or (episode >= DATASET_SIZE):
        timestr = time.strftime("%H:%M:%S", time.gmtime(time.time()-timer_start))
        log_str = 'Ep:{:07d}({:5.1f}%)  T:{:s}  avg.:{:f}'            .format(episode, episode/DATASET_SIZE*100,
                    timestr, eval_dist_AM_0.peek())
        logger.info(log_str)
        logger_start = time.time()
        
    if episode >= DATASET_SIZE:
        break

    
logger.info('---------------------------------------------------')
logger.info('average = {}'.format(eval_dist_AM_0.result()))
logger.info('---------------------------------------------------')
logger.info('---------------------------------------------------')


