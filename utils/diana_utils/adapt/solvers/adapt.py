# -*- coding: utf-8 -*-
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter
from .solver import register_solver
sys.path.append('../../')
# from utils.diana_utils.utils import ReverseLayerF, ConditionalEntropyLoss # circular import error
from pdb import set_trace
from time import time
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Function, Variable
from utils.loss_functions import *

import conf
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

class BaseSolver:
	"""
	Base DA solver class
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, query_count, device, args, run, tgt_sup_loader_wrong=None):
		self.net = net
		self.src_loader = src_loader
		self.tgt_sup_loader = tgt_sup_loader
		self.tgt_unsup_loader = tgt_unsup_loader
		self.train_idx = np.array(train_idx)
		self.tgt_opt = tgt_opt
		self.query_count = query_count
		self.device = device
		self.args = args
		self.run = run
		self.tgt_sup_loader_wrong = tgt_sup_loader_wrong

	
@register_solver('ft')
class TargetFTSolver(BaseSolver):
    #Finetune on target labels
    def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, query_count, device, args, run, tgt_sup_loader_wrong=None):
        super(TargetFTSolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, query_count, device, args, run, tgt_sup_loader_wrong)
    
    # Finetune on target labels
    def solve(self, epoch, writer, round_iter):
        self.net.train()		
        info_str = '[Train target finetuning] Epoch: {}'.format(epoch)
        
        if conf.args.turn_to_binary:
            raise NotImplementedError
        else:
            for data_t, target_t in self.tgt_sup_loader:
                data_t, target_t = data_t.to(self.device), target_t.to(self.device)
                round_iter += 1
                self.tgt_opt.zero_grad()
                output = self.net(data_t)
                loss = nn.CrossEntropyLoss()(output, target_t)
                info_str = '[Train target finetuning] Epoch: {}'.format(epoch)
                info_str += ' Target Sup. Loss: {:.3f}'.format(loss.item())
                loss.backward()
                self.tgt_opt.step()
        
        if epoch % 10 == 0: print(info_str)
        return round_iter

@register_solver('self_ft')
class SelfFTSolver(BaseSolver):
	"""
	Finetune on target labels
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, query_count, device, args, run, tgt_sup_loader_wrong=None,lab_model=None, lab_opt=None):
		super(SelfFTSolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, query_count, device, args, run, tgt_sup_loader_wrong)
		self.lab_model = lab_model
		self.lab_opt = lab_opt

	# for one adapting epoch 
	def solve_common_amp(self, epoch, writer, round_iter, uns_conf_loader, gmm1_loader, iter_num, scaler, gmm1_train=True, conf_mask=False, conf_mask_thres=0.95):  
		"""
		iter_item: target labeled, target conf data and src data seperately
		iter_num: decided by iter_num
		forward: together
		"""		
		self.net.train()
		sup_tgt_weight = 1.0
		src_weight, cc_weight, uc_weight = self.args['src_weight'], self.args['cc_weight'], self.args['uc_weight']

		if len(self.src_loader.sampler) == 0 or len(self.tgt_sup_loader.sampler) == 0 or len(uns_conf_loader) == 0:
			set_trace()
		if conf.args.turn_to_binary:
			if len(self.tgt_sup_loader_wrong.sampler) == 0:
				set_trace()
		if gmm1_train:
			assert len(gmm1_loader) > 0;gmm1_iter = iter(gmm1_loader)
		src_iter, tgtlab_iter, tgtconf_iter = iter(self.src_loader), iter(self.tgt_sup_loader), iter(uns_conf_loader) #64
		if conf.args.turn_to_binary:
			wrong_iter = iter(self.tgt_sup_loader_wrong)
		for batch_idx in range(iter_num):
			try:
				data_src, label_src = src_iter.next()
			except:
				src_iter = iter(self.src_loader)
				data_src, label_src = src_iter.next()
			
			try:
				data_ts, label_ts = tgtlab_iter.next()
			except:
				tgtlab_iter = iter(self.tgt_sup_loader)
				data_ts, label_ts = tgtlab_iter.next()
    
			if conf.args.turn_to_binary:
				try:
					data_ts_wrong, label_ts_wrong = wrong_iter.next()
				except:
					wrong_iter = iter(self.tgt_sup_loader)
					data_ts_wrong, label_ts_wrong = wrong_iter.next()

			try:
				data_w_conf, data_s_conf, label_tu_conf = tgtconf_iter.next()
			except:
				tgtconf_iter = iter(uns_conf_loader)
				data_w_conf, data_s_conf, label_tu_conf = tgtconf_iter.next()

			round_iter += 1
			info_str = '[Train with self_ft src_w:{}, tgt_w:{}, cc_w:{:.2f}] Epoch {} loss: '.format(src_weight, sup_tgt_weight , cc_weight, epoch)

			self.tgt_opt.zero_grad()  

			input_all = torch.cat([data_ts, data_src, data_w_conf, data_s_conf], dim=0)
    
			if gmm1_train:
				info_str = '[Train with self_ft src_w:{}, tgt_w:{}, cc_w:{:.2f}, uc_w:{}] Epoch {} loss: '.format(src_weight, sup_tgt_weight, cc_weight, uc_weight, epoch)
				try:
					data_w_g1, lab_w_g1  = gmm1_iter.next()
				except:
					gmm1_iter = iter(gmm1_loader)
					data_w_g1, lab_w_g1  = gmm1_iter.next()
				input_all = torch.cat([input_all, data_w_g1], dim=0)

			if conf.args.turn_to_binary:
				input_all = torch.cat([input_all, data_ts_wrong], dim =0)
    
			input_all = input_all.to(self.device)
			label_ts, label_src = label_ts.to(self.device), label_src.to(self.device)
			if conf.args.turn_to_binary:
				label_ts_wrong = label_ts_wrong.to(self.device)
			with autocast():
				output_all = self.net(input_all)
				output = output_all[:len(label_ts)]
				out_src =  output_all[len(label_ts):(len(label_ts)+len(data_src))]
				data_w_out, data_s_out = output_all[(len(label_ts)+len(data_src)):(len(label_ts)+len(data_src)+2*len(label_tu_conf))].chunk(2)
				if conf.args.turn_to_binary:
					output_wrong = output_all[-len(label_ts_wrong):]

					correct_loss = nn.CrossEntropyLoss()(output, label_ts)
					
					T_out_softmax = output_wrong.softmax(dim=1)
					filter_idx = (T_out_softmax > 1 / conf.args.opt['num_class'])
					for i in range(len(T_out_softmax)):
						filter_idx[i][label_ts_wrong[i]] = 0.0
					
					# T_wrong_targets = T_out_softmax.clone().detach()
					# T_wrong_targets[~filter_idx] = 0.0
					# T_wrong_targets = F.normalize(T_wrong_targets, p=1, dim=1)
					wrong_loss = conf.args.w_final_loss_wrong * complement_CrossEntropyLoss(output_wrong, label_ts_wrong)
					
					xeloss_tgt = correct_loss + wrong_loss
				else:
					xeloss_tgt = nn.CrossEntropyLoss()(output, label_ts)
					info_str += ' Target-Labeled ce_loss: {:.3f}'.format(xeloss_tgt.item()) # 		

				xeloss_src = nn.CrossEntropyLoss()(out_src, label_src)  #src_sup_wt
				info_str += " Source ce_loss: {:.3f}".format(xeloss_src.item())
				
				label_conf = torch.argmax(data_w_out.detach().clone(), dim=1).to(self.device)	
				uns_conf_self_loss = nn.CrossEntropyLoss()(data_s_out, label_conf)				
				info_str += ' Target-CC consistency_loss: {:.3f}'.format(uns_conf_self_loss.item())

				loss_ent = 0
				if gmm1_train:
					gmmp1_out = output_all[(len(label_ts)+len(data_src)+2*len(label_tu_conf)):(len(label_ts)+len(data_src)+2*len(label_tu_conf)+len(lab_w_g1))]
					gmmp1_probs = F.softmax(gmmp1_out, dim=1)
					loss_ent = - torch.mean(torch.sum(gmmp1_probs * 
													(torch.log(gmmp1_probs + 1e-5)), dim=1 ))		
					info_str += 'Target-UC entropy_loss: {:.3f}'.format(loss_ent)			

				total_loss = src_weight* xeloss_src + cc_weight * uns_conf_self_loss + sup_tgt_weight * xeloss_tgt + uc_weight * loss_ent   #没有unconf
			
			scaler.scale(total_loss).backward() 
			scaler.step(self.tgt_opt) 
			scaler.update()
			# if self.run == 0:  
			# 	writer.add_scalar('Run0/QueryCount-{}/FinetuneSrcLoss'.format(self.query_count), xeloss_src.item(), round_iter)
			# 	writer.add_scalar('Run0/QueryCount-{}/FinetuneTargetSupLoss'.format(self.query_count), xeloss_tgt.item(), round_iter)
			# 	writer.add_scalar('Run0/QueryCount-{}/TargetSelfLoss'.format(self.query_count), uns_conf_self_loss.item(), round_iter)
			# 	if gmm1_train: writer.add_scalar('Run0/QueryCount-{}/EntropyLoss'.format(self.query_count), loss_ent, round_iter)
		if epoch % 2 == 0: print(info_str)
		return round_iter


@register_solver('mme')
class MMESolver(BaseSolver):
	"""
	Implements MME from Semi-supervised Domain Adaptation via Minimax Entropy: https://arxiv.org/abs/1904.06487
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, query_count, device, args, run, tgt_sup_loader_wrong=None):
		super(MMESolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, query_count, device, args, run, tgt_sup_loader_wrong)

	def solve(self, epoch, writer, round_iter):
		"""
		Semisupervised adaptation via MME: XE on labeled source + XE on labeled target + \
										adversarial ent. minimization on unlabeled target
		"""
		self.net.train()		
		src_sup_wt, lambda_adent = self.args['src_sup_wt'], self.args['unsup_wt']  #, lambda_adent 1.0

		if self.query_count == 0:
			src_sup_wt, lambda_unsup = 1.0, 0.1
		else:
			src_sup_wt, lambda_unsup = self.args['src_sup_wt'], self.args['unsup_wt']  #src_sup_wt 0.1
			tgt_sup_iter = iter(self.tgt_sup_loader)
			if conf.args.turn_to_binary:
				tgt_sup_iter_wrong = iter(self.tgt_sup_loader_wrong)

		# print("-----mme src_sup_wt, lambda_adent:", src_sup_wt, lambda_adent)
		joint_loader = zip(self.src_loader, self.tgt_unsup_loader)
		for batch_idx, ((data_s, label_s), (data_tu, label_tu)) in enumerate(joint_loader):			
			data_s, label_s = data_s.to(self.device), label_s.to(self.device)
			data_tu = data_tu.to(self.device)
			
			if self.query_count > 0:
				try:
					data_ts, label_ts = next(tgt_sup_iter)
					data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
				except: break
				if conf.args.turn_to_binary:
					try:
						data_ts_wrong, label_ts_wrong = next(tgt_sup_iter_wrong)
						data_ts_wrong, label_ts_wrong = data_ts_wrong.to(self.device), label_ts_wrong.to(self.device)
					except: break
			round_iter += 1
			# zero gradients for optimizer
			self.tgt_opt.zero_grad()
					
			# log basic adapt train info
			info_str = "[Train Minimax Entropy] Epoch: {}".format(epoch)

			# extract features
			score_s = self.net(data_s)
			xeloss_src = src_sup_wt * nn.CrossEntropyLoss()(score_s, label_s)
			
			# log discriminator update info
			info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())
			
			xeloss_tgt = 0
			if self.query_count > 0:
				if conf.args.turn_to_binary:
					score_ts = self.net(data_ts)
					score_ts_wrong = self.net(data_ts_wrong)
					
					correct_loss = nn.CrossEntropyLoss()(score_ts, label_ts)
					
					# T_out_softmax = score_ts_wrong.softmax(dim=1)
					# filter_idx = (T_out_softmax > 1 / conf.args.opt['num_class'])
					# for i in range(len(T_out_softmax)):
					# 	filter_idx[i][label_ts_wrong[i]] = 0.0
					
					# T_wrong_targets = T_out_softmax.clone().detach()
					# T_wrong_targets[~filter_idx] = 0.0
					# T_wrong_targets = F.normalize(T_wrong_targets, p=1, dim=1)
					wrong_loss = conf.args.w_final_loss_wrong * complement_CrossEntropyLoss(score_ts_wrong, label_ts_wrong)
					
					xeloss_tgt = correct_loss + wrong_loss
					info_str += " Tgt Sup loss: {:.3f}".format(xeloss_tgt.item())
				else:
					score_ts = self.net(data_ts)
					xeloss_tgt = nn.CrossEntropyLoss()(score_ts, label_ts)
					info_str += " Tgt Sup loss: {:.3f}".format(xeloss_tgt.item())

			xeloss = xeloss_src + xeloss_tgt
			xeloss.backward()
			self.tgt_opt.step()

			# Add adversarial entropy
			self.tgt_opt.zero_grad()

			# score_tu = self.net(data_tu, reverse_grad=True)
			y_logit = self.net[1](self.net[0](data_tu), reverse_grad=True)
			score_tu = y_logit
   
			probs_tu = F.softmax(score_tu, dim=1)
			loss_adent = lambda_adent * torch.mean(torch.sum(probs_tu * (torch.log(probs_tu + 1e-5)), 1))
			loss_adent.backward()
			
			self.tgt_opt.step()
			
			# Log net update info
			info_str += " MME loss: {:.3f}".format(loss_adent.item())	
			
			# if self.run == 0 and round_iter % 10 ==0 and self.query_count % 2==0:  #warmup
			# 	writer.add_scalar('Run0/QueryCount-{}/SourceSupLoss'.format(self.query_count), xeloss_src.item(), round_iter)
			# 	writer.add_scalar('Run0/QueryCount-{}/TargetMMELoss'.format(self.query_count), loss_adent.item(), round_iter)	

		if epoch%10 == 0: print(info_str)
		return round_iter


@register_solver('dann')
class DANNSolver(BaseSolver):
	"""
	Implements DANN from Unsupervised Domain Adaptation by Backpropagation: https://arxiv.org/abs/1409.7495
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, query_count, device, args, run, tgt_sup_loader_wrong=None):
		super(DANNSolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, query_count, device, args, run, tgt_sup_loader_wrong)
	
	def solve(self, epoch, disc, disc_opt):
		"""
		Semisupervised adaptation via DANN: XE on labeled source + XE on labeled target + \
									ent. minimization on target + DANN on source<->target
		"""
		gan_criterion = nn.CrossEntropyLoss()
		cent = ConditionalEntropyLoss().to(self.device)

		self.net.train()
		disc.train()
		
		src_sup_wt, lambda_unsup, lambda_cent = 0.1, 1.0, 0.1  # self.args['src_sup_wt'], self.args['unsup_wt'], self.args['cent_wt']
		tgt_sup_iter = iter(self.tgt_sup_loader)

		joint_loader = zip(self.src_loader, self.tgt_unsup_loader)		
		for batch_idx, ((data_s, label_s), (data_tu, label_tu)) in enumerate(joint_loader):
			data_s, label_s = data_s.to(self.device), label_s.to(self.device)
			data_tu = data_tu.to(self.device)

			if self.query_count > 0:
				try:
					data_ts, label_ts = next(tgt_sup_iter)
					data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
				except: break

			# zero gradients for optimizers
			self.tgt_opt.zero_grad()
			disc_opt.zero_grad()

			# Train with target labels
			score_s = self.net(data_s)
			xeloss_src = src_sup_wt*nn.CrossEntropyLoss()(score_s, label_s)

			info_str = "[Train DANN] Epoch: {}".format(epoch)
			info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())                    

			xeloss_tgt = 0
			if self.query_count > 0:
				score_ts = self.net(data_ts)
				xeloss_tgt = nn.CrossEntropyLoss()(score_ts, label_ts)
				info_str += " Tgt Sup loss: {:.3f}".format(xeloss_tgt.item())

			# extract and concat features
			score_tu = self.net(data_tu)
			f = torch.cat((score_s, score_tu), 0)

			# predict with discriminator
			f_rev = ReverseLayerF.apply(f)
			pred_concat = disc(f_rev)

			target_dom_s = torch.ones(len(data_s)).long().to(self.device)
			target_dom_t = torch.zeros(len(data_tu)).long().to(self.device)
			label_concat = torch.cat((target_dom_s, target_dom_t), 0)

			# compute loss for disciminator
			loss_domain = gan_criterion(pred_concat, label_concat)
			loss_cent = cent(score_tu)

			loss_final = (xeloss_src + xeloss_tgt) + (lambda_unsup * loss_domain) + (lambda_cent * loss_cent)

			loss_final.backward()

			self.tgt_opt.step()
			disc_opt.step()
		
			# log net update info
			info_str += " DANN loss: {:.3f}".format(lambda_unsup * loss_domain.item())		
			info_str += " Ent Loss: {:.3f}".format(lambda_cent * loss_cent.item())		
		
		if epoch%2 == 0: print(info_str)


class ReverseLayerF(Function):
	"""
	Gradient negation utility class
	"""				 
	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg()
		return output, None

class ConditionalEntropyLoss(torch.nn.Module):
	"""
	Conditional entropy loss utility class
	"""				 
	def __init__(self):
		super(ConditionalEntropyLoss, self).__init__()

	def forward(self, x):
		b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
		b = b.sum(dim=1)
		return -1.0 * b.mean(dim=0)