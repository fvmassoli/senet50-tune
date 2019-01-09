from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#################
# Python imports
#################
import os
import PIL
import time
import threading
import numpy as np
from tqdm import tqdm as tq

##################
# Pytorch imports
##################
import torch.nn.functional as F

###############
# Tune imports
###############
import ray
from hyperopt import hp
from ray.tune.suggest import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.util import pin_in_object_store, get_pinned_object
from ray.tune import Trainable, run_experiments, register_trainable, Experiment

################
# local imports
################
from utils import *
from data_manager.data_manager import DataManager


pinned_obj_dict = {}


class TrainerClass(Trainable):
    def _setup(self, config):
        print("="*19)
        print("== Trials config ==")
        for k in self.config.keys():
            print("\t {}: {}".format(k, self.config[k]))
        print("="*19)
        self.data_loader_train = get_pinned_object(pinned_obj_dict['data_loader_train'])
        self.data_loader_valid = get_pinned_object(pinned_obj_dict['data_loader_valid'])
        self.args = get_pinned_object(pinned_obj_dict['args'])
        self.cuda_available = torch.cuda.is_available()
        print("Cuda is available: {}".format(self.cuda_available))
        self.model = get_model()
        if self.config['train_full_network'] == 0:
            for n, m in self.model.named_parameters():
                m.requires_grad = False
            self.model.classifier_1.weight.requires_grad = True
            self.model.classifier_1.bias.requires_grad = True
            nn.init.xavier_normal_(self.model.classifier_1.weight)
            self.model.classifier_1.bias.data.fill_(0)
        elif self.config['train_full_network'] == 1:
            for n, m in self.model.named_parameters():
                m.requires_grad = False
            self.model.classifier_1.weight.requires_grad = True
            self.model.classifier_1.bias.requires_grad = True
        else:
            pass
        if self.cuda_available:
            #if torch.cuda.device_count() > 1:
            #    print('========= Going multi GPU ==========')
            #    self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        opt = getattr(torch.optim, self.config['optimizer'])
        if self.config['optimizer'] == 'SGD':
            self.optimizer = opt(self.model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'],
                                 weight_decay=self.config['weight_decay'])
        else:
            self.optimizer = opt(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        self.batch_accumulation = self.config['batch_accumulation']

    def _train_iter(self):
        j = 1
        self.model.train()
        self.optimizer.zero_grad()
        #progress_bar = tq(self.data_loader_train)
        #progress_bar.set_description("Training")
 #       avg_loss = 0.0
        for batch_idx, (data, target, _) in enumerate(self.data_loader_train):
            if self.cuda_available:
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
#            avg_loss += loss.item()
            if j % self.batch_accumulation == 0:
                j = 1
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                j += 1
            #if batch_idx % self.args.logFrequency == 0:
            #    progress_bar.set_postfix({'Loss': '{:.3f}'.format(avg_loss/(batch_idx+1))})
        torch.cuda.empty_cache()
        # return avg_loss/len(self.data_loader_train)

    def _valid(self):
        self.model.eval()
        avg_loss = 0.0
        avg_acc = 0.0
        n_samples = 0
        #progress_bar = tq(self.data_loader_valid)
        #progress_bar.set_description("Validation")
        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(self.data_loader_valid):
                if self.cuda_available:
                    data = data.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                avg_loss += loss.item()
                y_hat = output.argmax(dim=1)
                avg_acc += (target == y_hat).sum().item()
                n_samples += len(target)
                if batch_idx % self.args.logFrequency == 0:
                    acc = avg_acc / n_samples
                    metrics = {
                        'loss': '{:.3f}'.format(avg_loss/(batch_idx+1)),
                        'acc': '{:.2f}%'.format(acc*100)
                    }
                    #progress_bar.set_postfix(metrics)
        loss = avg_loss / len(self.data_loader_valid)
        acc = avg_acc / n_samples
        print(metrics)
        torch.cuda.empty_cache()
        return {"loss": loss, "acc": acc}

    def _train(self):
        self._train_iter()
        return self._valid()

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(checkpoint_path)


def main(args):

    cuda_available = torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda_available:
        torch.cuda.manual_seed(args.seed)

    ray.init(num_gpus=1)

    ####################
    # Init data manager
    ####################
    data_manager = DataManager(train_folder=args.trainFolder,
                               valid_folder=args.validFolder,
                               train_batch_size=16,
                               valid_batch_size=16,
                               img_resolution=args.imageResolution,
                               interpolation_algo_name='bilinear',
                               interpolation_algo_val=PIL.Image.BILINEAR,
                               lowering_resolution_prob=args.lowerResolutionProb,
                               indices_step=args.indicesStep,
                               training_valid_split=args.trainValidSplit)

    train_data_loader, valid_data_loader = data_manager.get_data_loaders()
    pinned_obj_dict['data_loader_train'] = pin_in_object_store(train_data_loader)
    pinned_obj_dict['data_loader_valid'] = pin_in_object_store(valid_data_loader)
    #pinned_obj_dict['data_loader_valid_orig'] = pin_in_object_store(valid_data_loader_original_data)
    pinned_obj_dict['args'] = pin_in_object_store(args)

    trainable_name = 'hyp_search_train'
    register_trainable(trainable_name, TrainerClass)

    reward_attr = "acc"

    #############################
    # Define hyperband scheduler
    #############################
    hpb = AsyncHyperBandScheduler(time_attr="training_iteration",
                                  reward_attr=reward_attr,
                                  grace_period=10,
                                  max_t=300)

    ##############################
    # Define hyperopt search algo
    ##############################
    space = {
        'lr': hp.uniform('lr', 0.0005, 0.01),
        'momentum': hp.uniform('momentum', 0.1, 0.9),
        'optimizer': hp.choice('optimizer', ['SGD', 'Adam']),
        'weight_decay': hp.uniform('weight_decay', 1.e-5, 1.e-4),
        'batch_accumulation': hp.choice('batch_accumulation', [4, 8, 16, 32]),
        'train_full_network': hp.choice('train_full_network', [0, 1, 2])
    }
    hos = HyperOptSearch(space, max_concurrent=4, reward_attr=reward_attr)

    #####################
    # Define experiments
    #####################
    exp_name = "hyp_search_hyperband_hyperopt_{}".format(time.strftime("%Y-%m-%d_%H.%M.%S"))
    exp = Experiment(
        name=exp_name,
        run=trainable_name,
        num_samples=args.numSamples,  # the number of experiments
        resources_per_trial={
            "cpu": 4,
            "gpu": 1
        },
        checkpoint_freq=args.checkpointFreq,
        checkpoint_at_end=True,
        stop={
            reward_attr: 0.95,
            "training_iteration": args.trainingIteration,  # how many times a specific config will be trained
        },
        local_dir="./ray_results"
    )

    ##################
    # Run tensorboard
    ##################
    if args.runTensorBoard:
        thread = threading.Thread(target=launch_tensorboard, args=[exp_name])
        thread.start()
        launch_tensorboard(exp_name)

    ##################
    # Run experiments
    ##################
    run_experiments(exp, search_alg=hos, scheduler=hpb, verbose=True)


if __name__ == "__main__":
    main(get_args())
