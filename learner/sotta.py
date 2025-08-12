import conf
from utils import reset_utils
from utils.memory import FIFO, HUS, ConfFIFO, Uniform
from .dnn import DNN
from torch.utils.data import DataLoader
from utils.loss_functions import *
from utils.sam_optimizer import SAM, sam_collect_params


device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class SoTTA(DNN):
    def __init__(self, *args, **kwargs):
        super(SoTTA, self).__init__(*args, **kwargs)

    def init_learner(self):
        # turn on grad for BN params only
        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False
        for module in self.net.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    # With below, this module always uses the test batch statistics (no momentum)
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d):  # ablation study
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.LayerNorm):  # language models
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        if conf.args.sam:
            params, _ = sam_collect_params(self.net)
            optimizer = SAM(params, torch.optim.Adam, rho=0.05, lr=conf.args.opt['learning_rate'],
                                 weight_decay=conf.args.opt['weight_decay'])
        else:
            optimizer = torch.optim.SGD(
                self.net.parameters(),
                conf.args.opt['learning_rate'],
                momentum=conf.args.opt['momentum'],
                weight_decay=conf.args.opt['weight_decay'],
                nesterov=True)

        return optimizer

    def test_time_adaptation(self):
        assert isinstance(self.mem, FIFO) or isinstance(self.mem, ConfFIFO) \
               or isinstance(self.mem, Uniform) or isinstance(self.mem, HUS)
        if isinstance(self.mem, FIFO):
            feats, labels, _ = self.mem.get_memory()
        elif isinstance(self.mem, Uniform) or isinstance(self.mem, ConfFIFO) or isinstance(self.mem, HUS):
            feats, _, _, _, labels = self.mem.get_memory()
        else:
            raise AssertionError

        if len(feats) == 0:
            return
        feats = torch.stack(feats).to(device)
        labels = torch.Tensor(labels).type(torch.long).to(device)

        dataset = torch.utils.data.TensorDataset(feats, labels)
        data_loader = DataLoader(dataset, batch_size=conf.args.update_every_x,
                                 shuffle=True, drop_last=False, pin_memory=False)

        for e in range(conf.args.epoch):
            for batch_idx, (feats, _) in enumerate(data_loader):
                if len(feats) == 1:
                    self.net.eval()  # avoid BN error
                else:
                    self.net.train()

                loss_fn = HLoss(conf.args.temperature)
                
                feats = feats.to(device)
                preds_of_data = self.net(feats)

                loss_first = loss_fn(preds_of_data)

                if conf.args.enable_bitta:
                    loss_first += self.get_bitta_ssl_loss()
                    
                self.optimizer.zero_grad()

                loss_first.backward()

                if not isinstance(self.optimizer, SAM):
                    self.optimizer.step()
                else:
                    # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
                    self.optimizer.first_step(zero_grad=True)

                    preds_of_data = self.net(feats)

                    # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
                    loss_second = loss_fn(preds_of_data)
                    
                    if conf.args.enable_bitta:
                        loss_second += self.get_bitta_ssl_loss()

                    loss_second.backward()

                    self.optimizer.second_step(zero_grad=False)
                    