# -*- coding: utf-8 -*-

import sys
import argparse
import random

import math
import numpy as np
import torch
import time
import os
import conf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import webdataset as wds

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# For results reproducibility; would increase GPU memory ~24MiB
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_path():
    path = conf.args.log_name

    # information about used data type
    path += conf.args.dataset + '/'

    # information about used model type

    path += conf.args.method + '/'

    # information about domain(condition) of training data
    if conf.args.src == ['rest']:
        path += 'src_rest' + '/'
    elif conf.args.src == ['all']:
        path += 'src_all' + '/'
    elif conf.args.src is not None and len(conf.args.src) >= 1:
        path += 'src_' + '_'.join(conf.args.src) + '/'

    if conf.args.tgt:
        path += 'tgt_' + conf.args.tgt + '/'

    path += conf.args.log_prefix + '/'

    checkpoint_path = path + 'cp/'
    log_path = path
    result_path = path + '/'

    print('Path:{}'.format(path))
    return result_path, checkpoint_path, log_path


def main():
    ######################################################################
    device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    ################### Hyperparameters #################
    if 'cifar100' in conf.args.dataset:
        opt = conf.CIFAR100Opt
    elif 'cifar10' in conf.args.dataset:
        opt = conf.CIFAR10Opt
    elif 'tiny-imagenet' in conf.args.dataset:
        opt = conf.TINYIMAGENET_C
    elif 'imagenetR' in conf.args.dataset:
        opt = conf.IMAGENET_R
    elif 'imagenet' in conf.args.dataset:
        opt = conf.IMAGENET_C
    elif 'pacs' in conf.args.dataset:
        opt = conf.PACSOpt
    elif 'vlcs' in conf.args.dataset:
        opt = conf.VLCSOpt
    elif 'office_home' in conf.args.dataset:
        opt = conf.OfficeHomeOpt
    elif 'domainnet-126' in conf.args.dataset:
        opt = conf.DOMAINNET126Opt
    elif 'colored-mnist' in conf.args.dataset:
        opt = conf.COLORED_MNIST
    else:
        raise NotImplementedError


    if conf.args.lr:
        if conf.args.memory_size == 1:
            conf.args.lr /= 64
        opt['learning_rate'] = conf.args.lr
        
    if conf.args.weight_decay:
        opt['weight_decay'] = conf.args.weight_decay
    if conf.args.dataloader_batch_size:
        opt['batch_size'] = conf.args.dataloader_batch_size
    conf.args.opt = opt

    model = None
    tokenizer = None  # for language models

    if conf.args.model == "resnet18_pretrained":
        import torchvision
        model = torchvision.models.resnet18
    elif conf.args.model == "resnet18":
        from models import ResNet
        model = ResNet.ResNet18
    elif conf.args.model == "resnet50_pretrained":
        import torchvision
        model = torchvision.models.resnet50
    elif conf.args.model == "resnet50":
        from models import ResNet
        model = ResNet.ResNet50
    elif conf.args.model == "resnet50_domainnet":
        from models import ResNet
        model = ResNet.ResNet50_DOMAINNET
    elif conf.args.model == "vitbase16_pretrained":
        import torchvision
        model = torchvision.models.vit_b_16
    elif conf.args.model == "vitbase16":
        from models.ViT import vit_b_16
        model = vit_b_16
    else:
        raise NotImplementedError

    # import modules after setting the seed
    from data_loader import data_loader as data_loader
    from learner.dnn import DNN
    from learner.bn_stats import BN_Stats
    from learner.tent import TENT
    from learner.sotta import SoTTA
    from learner.cotta import CoTTA
    from learner.sar import SAR
    from learner.rotta import RoTTA
    from learner.eata import ETA, EATA
    from learner.bitta import BiTTA

    from learner.simatta import SimATTA
    from learner.simatta_bin import SimATTA_BIN

    result_path, checkpoint_path, log_path = get_path()

    ########## Dataset loading ############################
    # source model
    if conf.args.method == 'Src':
        learner_method = DNN

    # TTA baselines
    elif conf.args.method == 'BN_Stats':
        learner_method = BN_Stats
    elif conf.args.method == 'TENT':
        learner_method = TENT
    elif conf.args.method == 'CoTTA':
        learner_method = CoTTA
    elif conf.args.method == 'SAR':
        learner_method = SAR
    elif conf.args.method == "RoTTA":
        learner_method = RoTTA
    elif conf.args.method == 'SoTTA':
        learner_method = SoTTA
    elif conf.args.method == "EATA":
        learner_method = EATA
    elif conf.args.method == "ETA":
        learner_method = ETA

    # ADA baselines
    elif conf.args.method == "ELPT":
        learner_method = ELPT
    elif conf.args.method == "MHPL":
        learner_method = MHPL
    elif conf.args.method == "EADA":
        learner_method = EADA
    elif conf.args.method in ["CLUE", "DIANA"]:
        learner_method = DIANA

    # Active TTA baselines
    elif conf.args.method == "SimATTA":
        learner_method = SimATTA
    elif conf.args.method == "SimATTA_BIN":
        learner_method = SimATTA_BIN

    # Our method
    elif conf.args.method == "BiTTA":
        learner_method = BiTTA
    else:
        raise NotImplementedError

    corruption_list = []

    # modify for continual adaptation
    if conf.args.tgt == "cont":
        if conf.args.dataset == "pacs":
            cont_seq = conf.CONT_SEQUENCE_PACS
        elif conf.args.dataset == "vlcs":
            cont_seq = conf.CONT_SEQUENCE_VLCS
        elif conf.args.dataset == "domainnet-126":
            cont_seq = conf.CONT_SEQUENCE_DOMAINNET126
        else:
            cont_seq = conf.CONT_SEQUENCE_TINYIMAGENET


        if conf.args.cont_seq not in cont_seq.keys():
            corruption_list_ = cont_seq[0]
            random.shuffle(corruption_list_)
        else:
            corruption_list_ = cont_seq[conf.args.cont_seq]
        
        if conf.args.dataset == 'tiny-imagenet':
            # Note that ATTA paper removed brightness-5 here because they fine-tuned the model with brightness;
            # we pretrain with tiny-imagenet and maintain brightness-5 here
            pass
        corruption_list += corruption_list_
        
    else:
        corruption_list += [conf.args.tgt]


    original_result_path, original_checkpoint_path, original_log_path = result_path, checkpoint_path, log_path
    learner = learner_method(model, corruption_list)
    
    if conf.args.random_setting:
        assert conf.args.tgt == "cont"
        corruption_list = [corruption_list]

    for corruption in corruption_list:
        learner.temp_value = 0
        if conf.args.random_setting:
            corruption_name = "random"
        else:
            corruption_name = corruption
            
        if conf.args.tgt == "cont":
            result_path = original_result_path + corruption_name + "/"
            checkpoint_path = original_checkpoint_path + corruption_name + "/"
            log_path = original_log_path + corruption_name + "/"

        else:
            result_path = original_result_path
            checkpoint_path = original_checkpoint_path
            log_path = original_log_path

        learner.init_json(log_path)
        learner.occurred_class = [0 for i in range(conf.args.opt['num_class'])]

        if conf.args.reset_every_corruption:
            learner.reset()
            
        since = time.time()
        
        if conf.args.wds_path is None:
            print('##############Source Data Loading...##############')
            set_seed()  # reproducibility
            if conf.args.dataset not in ["imagenetR", "colored-mnist"]:
                source_train_data_loader, source_val_data_loader = data_loader.domain_data_loader(conf.args.dataset,
                                                                                                conf.args.src,
                                                                                                conf.args.opt['file_path'],
                                                                                                batch_size=conf.args.opt[
                                                                                                    'batch_size'],
                                                                                                valid_split=0,
                                                                                                # valid_split=0.3,
                                                                                                # to be used for the validation
                                                                                                test_split=0, is_src=True,
                                                                                                num_source=conf.args.num_source)
            else:
                source_train_data_loader = None
                source_val_data_loader = None


            
                
            print('##############Target Data Loading...##############')
            set_seed()  # reproducibility
            target_data_loader, _ = data_loader.domain_data_loader(conf.args.dataset, corruption,
                                                                conf.args.opt['file_path'],
                                                                batch_size=conf.args.opt['batch_size'],
                                                                valid_split=0,
                                                                test_split=0, is_src=False,
                                                                num_source=conf.args.num_source)

            set_seed()  # reproducibility
                
            learner.set_target_data(source_train_data_loader, source_val_data_loader, target_data_loader, corruption_name)
        
        else:
            if isinstance(corruption, list) and len(corruption) > 10:
                str_cor = "random"
            else:
                str_cor = str(corruption)
            url = os.path.join(conf.args.wds_path, f"{conf.args.dataset}_{conf.args.seed}_dist1_{str_cor}.tar")
            import torchvision

            preproc = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            )
            dataset = (
                wds.WebDataset(url)
                .decode("pil")
                .to_tuple("input.jpg", "output.cls")
                .map_tuple(preproc, lambda x : x)
            )
            dataloader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=conf.args.opt['batch_size'])
            learner.iter_target_train_set = iter(dataloader)
            learner.target_train_set = None
        
        time_elapsed = time.time() - since
        print('Data Loading Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        #################### Training #########################     

        since = time.time()
        
        learner.run_before_training()

        # make dir if it doesn't exist
        if not os.path.exists(result_path):
            oldumask = os.umask(0)
            os.makedirs(result_path, 0o777)
            os.umask(oldumask)
        if not os.path.exists(checkpoint_path):
            oldumask = os.umask(0)
            os.makedirs(checkpoint_path, 0o777)
            os.umask(oldumask)
        script = ' '.join(sys.argv[1:])

        set_seed()  # reproducibility

        if conf.args.online == False:
            start_epoch = 1
            best_acc = -9999
            best_epoch = -1

            for epoch in range(start_epoch, conf.args.epoch + 1):
                learner.train(epoch)

            learner.save_checkpoint(epoch=0, epoch_acc=-1, best_acc=best_acc,
                                    checkpoint_path=checkpoint_path + 'cp_last.pth.tar')
            learner.dump_eval_online_result(is_train_offline=True)  # eval with final model

            time_elapsed = time.time() - since
            print('Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))

        elif conf.args.online == True:

            current_num_sample = 1
            num_sample_end = conf.args.nsample
            best_acc = -9999
            best_epoch = -1

            TRAINED = 0
            SKIPPED = 1
            FINISHED = 2

            finished = False

            while not finished and current_num_sample < num_sample_end:

                ret_val = learner.train_online(current_num_sample)

                if ret_val == FINISHED:
                    break
                elif ret_val == SKIPPED:
                    pass
                elif ret_val == TRAINED:
                    pass
                current_num_sample += 1

            if not conf.args.remove_cp:
                learner.save_checkpoint(epoch=0, epoch_acc=-1, best_acc=best_acc,
                                        checkpoint_path=checkpoint_path + 'cp_last.pth.tar')
            learner.dump_eval_online_result()

            if conf.args.wandb:
                import wandb
                wandb.log({
                    "num_batch_adapt": learner.num_batch_adapt,
                    "corruption_acc": {
                        str(corruption): learner.json_eval["accuracy"][-1]}
                })

            time_elapsed = time.time() - since
            print('Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))

        if conf.args.remove_cp:
            last_path = checkpoint_path + 'cp_last.pth.tar'
            best_path = checkpoint_path + 'cp_best.pth.tar'
            try:
                os.remove(last_path)
                os.remove(best_path)
            except Exception as e:
                pass

def parse_arguments(argv):
    """Command line parse."""

    # Note that 'type=bool' args should be False in default. Any string argument is recognized as "True". Do not give "--bool_arg 0"

    parser = argparse.ArgumentParser()

    ###MANDATORY###
    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset to be used, in [ichar, icsr, dsa, hhar, opportunity, gait, pamap2]')

    parser.add_argument('--model', type=str, default='HHAR_model',
                        help='Which model to use')

    parser.add_argument('--method', type=str, default='',
                        help='specify the method name')

    parser.add_argument('--src', nargs='*', default=None,
                        help='Specify source domains; not passing an arg load default src domains specified in conf.py')
    parser.add_argument('--tgt', type=str, default=None,
                        help='specific target domain; give "src" if you test under src domain')
    parser.add_argument('--gpu_idx', type=int, default=0, help='which gpu to use')

    ###Optional###
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate to overwrite conf.py')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='weight_decay to overwrite conf.py')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--epoch', type=int, default=1,
                        help='How many epochs do you want to use for train')
    parser.add_argument('--load_checkpoint_path', type=str, default='',
                        help='Load checkpoint and train from checkpoint in path?')
    parser.add_argument('--train_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for train')
    parser.add_argument('--valid_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for valid')
    parser.add_argument('--test_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for test')
    parser.add_argument('--nsample', type=int, default=20000000,
                        help='How many samples do you want use for train')
    parser.add_argument('--log_prefix', type=str, default='',
                        help='Prefix of log file path')
    parser.add_argument('--remove_cp', action='store_true',
                        help='Remove checkpoints after evaluation')
    parser.add_argument('--data_gen', action='store_true',
                        help='generate training data with source for training estimator')

    parser.add_argument('--num_source', type=int, default=100,
                        help='number of available sources')
    parser.add_argument('--parallel', type=bool, default=False)

    parser.add_argument('--log_name', type=str, default='log/')
    parser.add_argument('--wandb', type=str, default="", help="wandb project name. Leave empty if none")
    parser.add_argument('--wandb_name', type=str, default="", help="wandb run name. Leave empty if none")
    parser.add_argument('--cont_seq', default=0, type=int, help='switch to various order of cont datastream')

    # Distribution
    parser.add_argument('--tgt_train_dist', type=int, default=1,
                        help='0: real selection'
                             '1: random selection'
                        )
    parser.add_argument('--online', action='store_true', help='training via online learning?')
    parser.add_argument('--update_every_x', type=int, default=1, help='number of target samples used for every update')
    parser.add_argument('--memory_size', type=int, default=1,
                        help='number of previously trained data to be used for training')
    parser.add_argument('--memory_type', type=str, default='FIFO', help='FIFO')

    # CoTTA
    parser.add_argument('--ema_factor', type=float, default=0.999,
                        help='hyperparam for CoTTA')
    parser.add_argument('--restoration_factor', type=float, default=0.0,
                        help='hyperparam for CoTTA')
    parser.add_argument('--aug_threshold', type=float, default=0.92,
                        help='hyperparam for CoTTA')

    # SoTTA
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='momentum')
    parser.add_argument('--use_learned_stats', action='store_true', help='Use learned stats')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature for HLoss')
    parser.add_argument('--loss_scaler', type=float, default=0,
                        help='loss_scaler for entropy_loss')
    parser.add_argument('--validation', action='store_true',
                        help='Use validation data instead of test data for hyperparameter tuning')
    parser.add_argument('--adapt_then_eval', action='store_true',
                        help='Evaluation after adaptation - unrealistic and causing additoinal latency, but common in TTA.')
    parser.add_argument('--no_adapt', action='store_true', help='no adaptation')
    parser.add_argument('--skip_thres', type=int, default=1,
                        help='skip threshold to discard adjustment')

    parser.add_argument('--dummy', action='store_true', default=False, help='do nothing')

    # SAM (currently only supports SoTTA)
    parser.add_argument('--sam', action='store_true', default=False, help='changes Adam to SAM-Adam')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int,
                        help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000.,
                        help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000) * 0.40,
                        help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05,
                        help='\epsilon in Eqn. (5) for filtering redundant samples')

    parser.add_argument('--high_threshold', default=0.99, type=float, help='High confidence threshold')
    
    parser.add_argument('--num_eval_source', default=1000, type=int, help='number of source data used for SrcValidation')

    parser.add_argument('--turn_off_reset', action='store_true', default=False,
                        help="turn off default resetting algorithm, ie SAR and CoTTA")
    
    # SimATTA
    parser.add_argument("--early_stop", action='store_true', default=False)
    parser.add_argument("--weight_cluster_by_entropy", action='store_true', default=False)
    parser.add_argument("--atta_cold_start", type=int, default=100, help='')
    parser.add_argument("--atta_budget", type=int, default=100000)
    parser.add_argument("--atta_inc_rate", type=float, default=1.0)
    parser.add_argument("--atta_upper_th", type=float, default=0.01)
    parser.add_argument("--atta_lower_th", type=float, default=0.0001)
    parser.add_argument("--start_num_cluster", type=int, default=10)
    parser.add_argument('--atta_limit_mem_size' , type=int, default=None)
    parser.add_argument('--atta_limit_batch_active_sample' , type=int, default=None)
    # adaptation batch size : SimATTA has batch size smaller than memory's size in test time
    parser.add_argument('--atta_batch_size', type=int, default=64, help='')
    
    # TTA with binary feedback
    parser.add_argument('--n_active_sample', type=int, default=3, help='number of active samples per batch')
    parser.add_argument('--active_sample_per_n_step', type=int, default=0, help='test sparse active sampling with only sampling 1 batch per N steps')
    parser.add_argument("--active_full_label", action='store_true', default=False)
    parser.add_argument('--feedback_error_rate', type=float, default=0, help='Feedback error rate')
    parser.add_argument('--enable_bitta',action='store_true', default=False, help="Use to turn on TTA with binary feedback")

    # BiTTA
    parser.add_argument('--w_final_bfa_loss', type=float, default=1.0, help='')
    parser.add_argument('--w_final_aba_loss', type=float, default=1.0, help='')
    parser.add_argument('--sample_selection', type=str, default="random")

    parser.add_argument('--n_dropouts', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=-1)

    parser.add_argument('--reset_every_corruption', action='store_true', default=False, help="Reset the model and memory every corruption")


    # random setting
    parser.add_argument('--random_setting', action='store_true', default=False)
    parser.add_argument('--dataloader_batch_size',type=int, default=None, help='')

    # enhance tta
    parser.add_argument('--enhance_tta',action='store_true', default=False)
    parser.add_argument('--enhance_tta_lr',type=float, default=0.001, help='')
    parser.add_argument('--enhance_tta_epoch',type=int, default=10, help='')
    parser.add_argument('--enhance_tta_batchsize',type=int, default=64, help='')
    parser.add_argument('--enhance_save_path',type=str, default="enhance_cp", help='')

    parser.add_argument('--save_wds_dataset',action='store_true', default=False)
    parser.add_argument('--wds_path',type=str, default=None, help='')
    
    parser.add_argument('--vit_patch_size',type=int, default=16, help='')
    parser.add_argument('--replace_augmentation',action='store_true', default=False)
    parser.add_argument('--replace_entropy_loss',action='store_true', default=False)
    parser.add_argument('--use_original_conf',action='store_true', default=False)

    parser.add_argument('--dirichlet_beta', type=float, default=0.1,
                        help='the concentration parameter of the Dirichlet distribution for heterogeneous partition.')
    
    parser.add_argument('--enable_skip', type=int, default=0, help='')
    parser.add_argument('--feedback_delay', type=int, default=0, help='number of batches to delay the feedback samples')
    parser.add_argument('--enable_mecta',action='store_true', default=False, help='enable mecta norm')
    parser.add_argument('--direct_minimization',action='store_true', default=False, help='enable mecta norm')
     
    return parser.parse_args()


def set_seed():
    torch.manual_seed(conf.args.seed)
    np.random.seed(conf.args.seed)
    random.seed(conf.args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    print('Command:', end='\t')
    print(" ".join(sys.argv))
    conf.args = parse_arguments(sys.argv[1:])
    print(conf.args)
    set_seed()

    if conf.args.wandb:
        import wandb
        wandb.init(project=conf.args.wandb, name=conf.args.method+"/"+conf.args.wandb_name, config=vars(conf.args))
        wandb.define_metric("current_accuracy", summary="mean")
        wandb.define_metric("original_mean_conf_gt_class", summary="mean")
        wandb.define_metric("dropout_mean_conf_gt_class", summary="mean")
        wandb.define_metric("wall_clock_time_per_batch", summary="mean")

    main()
