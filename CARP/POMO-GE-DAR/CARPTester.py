
import torch

from logging import getLogger

from torch.utils.data import DataLoader


from CARPEnv import CARPEnv as Env
# from CARPTestEnv import CARPEnv as Env
from CARPModel import CARPModel as Model

from CARProblemDef import CARPDataset
from CARPTestData import CARPTestDataset
from utils.utils import *


class CARPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(device, **self.env_params)
        self.model = Model(**self.model_params)


        self.test_data = CARPTestDataset(self.tester_params['test_episodes'],
                                         self.env_params['vertex_size'],
                                         self.env_params['edge_size'],
                                         device,
                                         10,
                                         10
                                         )

        if self.tester_params['test_data_load']['enable']:
            self.test_data.Dataload(self.tester_params['test_data_load']['filename'])
        else:
            self.test_data.DataGenerate()


        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        print("checkpoint: ",checkpoint_fullname)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        t1 = time.time()
        score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        batch_size = self.tester_params['test_batch_size']
        test_loader = DataLoader(self.test_data, batch_size, False, num_workers=0, generator = torch.Generator(device='cuda'))
        for batch_idx, batch in enumerate(test_loader):

            score = self._test_one_batch(batch_size, batch)

            score_AM.update(score, batch_size)
            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            print("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score ))

            all_done = (episode == test_num_episode)

            if all_done:
                print(" *** Test Done *** ")
                print(" SCORE: {:.4f} ".format(score_AM.avg))
                print("Test time: {}".format(time.time() - t1))
        return score_AM.avg


    def _test_one_batch(self, batch_size, batch):


        # Ready
        ###############################################

        self.model.eval()

        with torch.no_grad():
            self.env.load_problems(batch_size, batch)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            # state, reward, done, route = self.env.step(selected)
            state, reward, done = self.env.step(selected)

        # Return
        ###############################################
        aug_reward = reward.reshape(1, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, max_indices = aug_reward.max(dim=2)  # get best results from pomo

        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value
        return no_aug_score.item()
