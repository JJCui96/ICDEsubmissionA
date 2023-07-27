from BaseModel import BaseModel
import torch.nn as nn
import torch
import os
import importlib
from copy import deepcopy
from utils import *
from models import DKT, SAKT, QTE, AKT
import torch.nn.functional as F
import tqdm

class DKTwithStateFetcher(DKT.DKT):

	def get_state(self, feed_dict):

		self.layers.flatten_parameters()

		knows = feed_dict['knows']		# B, S, K
		corrs = feed_dict['corrs']		# B, S

		bias_knows = (knows > 0)*(knows + (knows > 0)*\
			(corrs.unsqueeze(-1))*(self.args.n_knows + 1))		# B, S, K
		inputs = self.knows_embedding(bias_knows).sum(-2)		# B, S, D

		output, (_, _) = self.layers(inputs)					# B, S, D
		return output[:, :-1]

class SAKTwithStateFetcher(SAKT.SAKT):

	def get_state(self, feed_dict):

		knows = feed_dict['knows']				# B, S, K
		corrs = feed_dict['corrs']				# B, S
		(B, S, K), D = knows.size(), self.args.d_hidden
		
		bias_knows = (knows > 0)*(knows + (knows > 0)*\
			(corrs.unsqueeze(-1))*(self.args.n_knows + 1))		# B, S, K
		inter_emb = self.inter_embedding(bias_knows).sum(-2)	# B, S, D
		knows_emb = self.knows_embedding(knows).sum(-2)			# B, S, D
		pos = torch.arange(S).to(self.args.device).repeat(B, 1)	# B, S
		pos_emb = self.pos_embedding(pos)

		kv_input = inter_emb + pos_emb
		q = knows_emb[:, 1:]										# B, S, D
		k = kv_input[:, :-1]										# B, S, D
		v = k													# B, S, D
		mask = torch.ones(S - 1, S - 1).to(self.args.device).triu(1)# S, S
		output, _ = self.attn(q, k, v, attn_mask = mask.bool())	# B, S, D
		output = self.layer_norm1(output + q)					# B, S, D
		output = self.layer_norm2(output + self.FFN(output))	# B, S, D

		return output

class AKTwithStateFetcher(AKT.AKT):

	def get_state(self, feed_dict):

		knows = feed_dict['knows']
		corrs = feed_dict['corrs']							# bs, sl
		pid_data = feed_dict['probs']

		bias_knows = (knows > 0)*(knows + corrs.unsqueeze(-1)*(self.args.n_knows + 1))
		q_data, qa_data = knows, bias_knows

		q_embed_data = self.q_embed(q_data).sum(-2)  # BS, seqlen,  d_model# c_ct
		qa_embed_data = self.qa_embed(qa_data).sum(-2)

		q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct
		q_embed_diff_data = q_embed_diff_data.sum(-2)
		pid_embed_data = self.difficult_param(pid_data)  # uq
		q_embed_data = q_embed_data + pid_embed_data * \
			q_embed_diff_data  # uq *d_ct + c_ct
		
		qa_embed_diff_data = self.qa_embed_diff(qa_data)
		qa_embed_diff_data = qa_embed_diff_data.sum(-2)

		qa_embed_data = qa_embed_data + pid_embed_data * \
			qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
		
		c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2

		# BS.seqlen,d_model
		# Pass to the decoder
		# output shape BS,seqlen,d_model or d_model//2
		d_output = self.model(q_embed_data, qa_embed_data)  # 211x512

		return d_output[:, 1:]


class DR4KT(BaseModel):

	@staticmethod
	def parse_args(parser):
		parser.add_argument('--temp', type = float, default = 1,
			help = 'Temperature value.')
		parser.add_argument('--lamb', type = float, default = 1,
			help = 'Loss fusion value.')
		parser.add_argument('--emb_model', type = str, default = 'DKT',
			help = 'The embedded model in DR4KT.')
		parser.add_argument('--dr_do', type = float, default = 0,
			help = 'Dropout in DR4KT.')
		parser.add_argument('--dr_lr', type = float, default = 0.001,
			help = 'Dropout in DR4KT.')
		parser.add_argument('--ori_l2', type = float, default = 0.00,
			help = 'Dropout in DR4KT.')
		parser.add_argument('--dr_l2', type = float, default = 0.00,
			help = 'Dropout in DR4KT.')
		args, _ = parser.parse_known_args()
		model_class = eval('{}withStateFetcher'.format(args.emb_model))
		model_class.parse_args(parser)
	
	def __init__(self, args):
		super().__init__(args)
		model_class = eval('{}withStateFetcher'.format(args.emb_model))
		self.emb_model = model_class(self.args).to(self.args.device)
		self.fuser_gen = nn.Sequential(
			nn.Linear(args.d_hidden + 1, args.d_hidden//2),
			nn.ReLU(),
			nn.Linear(args.d_hidden//2, 1),
			nn.Sigmoid()
		)

	def get_optimizer(self, step = 0):

		ori_paras = list()
		dr_paras = list()

		for name, para in self.named_parameters():
			if 'fuser_gen' in name:
				dr_paras.append(para)
			else:
				ori_paras.append(para)
		
		if step == 0:
			return torch.optim.Adam(ori_paras, lr = self.args.lr, weight_decay = self.args.ori_l2)
		if step == 1:
			return torch.optim.Adam(dr_paras, lr = self.args.dr_lr, weight_decay = self.args.dr_l2)


	def get_feed_dict(self, batch, mode):
		return self.emb_model.get_feed_dict(batch, mode)
	
	def loss(self, feed_dict):
		scores = feed_dict['scores']
		kt_scores = feed_dict['unbias_scores']
		labels = feed_dict['labels']
		weights = feed_dict['weights']
		kt_loss = F.binary_cross_entropy(kt_scores, labels.float(), reduction = 'none')
		kt_loss = (kt_loss*weights).mean()

		pred_loss = F.binary_cross_entropy(scores, labels.float())
		
		if feed_dict['step'] == 0:
			if self.args.emb_model == 'AKT':
				kt_loss += feed_dict['pl2_loss']
			return kt_loss
		if feed_dict['step'] == 1:
			return pred_loss


	def forward(self, feed_dict):


		self.emb_model.forward(feed_dict)
		unbias_scores = feed_dict['ori_scores']
		self.qte.forward(feed_dict)
		tends = feed_dict['ori_scores'].detach()


		knows = feed_dict['knows']
		probs = feed_dict['probs']
		corrs = feed_dict['corrs']



		qte_emb = self.qte.get_emb(knows, probs).detach()# B, S, D
		state = self.emb_model.get_state(feed_dict)		# B, S-1, D
		discs = corrs*(1 - tends) + (~corrs)*tends

		weights = (discs.log()/self.args.temp).exp()

		qte_emb = qte_emb[:, 1:]
		weights = weights[:, 1:]
		tends = tends[:, 1:]
		
		diff = (unbias_scores - tends).abs().unsqueeze(-1)

		fuser_gen_input = torch.cat([diff*state, diff*unbias_scores.unsqueeze(-1)], -1) + \
				   torch.cat([(1 - diff)*qte_emb, (1 - diff)*tends.unsqueeze(-1)], -1)
		scores = self.fuser_gen(fuser_gen_input).squeeze(-1)	# B, S-1


		feed_dict['unbias_scores'] = unbias_scores[feed_dict['filt']]
		feed_dict['weights'] = weights[feed_dict['filt']]
		feed_dict['scores'] = scores[feed_dict['filt']]
		feed_dict['ori_scores'] = scores
		



	def train_an_epoch(self, data):	
		self.train()
		train_loss = list()
		random.shuffle(data)
		if self.args.bar:
			dataset_bar = tqdm.tqdm(range(0, len(data), self.args.batch_size), 
				leave = False, ncols = 50, mininterval = 1)
		else:
			dataset_bar = range(0, len(data), self.args.batch_size)
		for i in dataset_bar:
			self.optim0.zero_grad()
			batch = data[i:i + self.args.batch_size]
			feed_dict = self.get_feed_dict(batch, 'train')
			feed_dict['step'] = 0
			self.forward(feed_dict)
			loss = self.loss(feed_dict)
			train_loss.append(loss.item())
			loss.backward()
			self.optim0.step()

			self.optim1.zero_grad()
			feed_dict['step'] = 1
			self.forward(feed_dict)
			loss = self.loss(feed_dict)
			loss.backward()
			self.optim1.step()

		train_loss = np.mean(train_loss)
		return train_loss
		
	
	def fit(self, train_set, valid_set, order):
		
			

		file_path = 'log/' + self.args.dataset + '/QTE/' + str(order) + '.mdl'
		self.qte = torch.load(file_path).to(self.args.device)

		self.optim0 = self.get_optimizer(0)
		self.optim1 = self.get_optimizer(1)

		valid_metric = float('-inf')
		stop_counter = 0
				
		eval_dict = self.eval_an_epoch(valid_set)
		log = ''
		for key in eval_dict:
			log += '{}: {:.4f}, '.format(key, eval_dict[key])
		print('Evaluation before training:', end = ' ')
		print(log)

		best_model = deepcopy(self)
		best_eval = deepcopy(eval_dict)

		try:
			for epo in range(self.args.epoch):
				print('Epoch {0:03d}'.format(epo), end = ' ')
				train_loss = self.train_an_epoch(train_set)
				eval_dict = self.eval_an_epoch(valid_set)
				log = 'train_loss: {:.4f}, '.format(train_loss)
				log += performance_str(eval_dict)
				print(log, end = ' ')
				metric_total = get_performance(eval_dict)
				if metric_total > valid_metric:
					valid_metric = metric_total
					best_model = deepcopy(self)
					best_eval = deepcopy(eval_dict)
					stop_counter = 0
					print('* {:.4f}'.format(metric_total), end = '')
				else:
					stop_counter += 1
				print()
				if stop_counter == self.args.early_stop or epo == self.args.epoch - 1:
					print('Training stopped.')
					print('valid:\t', performance_str(best_eval))
					if self.args.save_model:
						file_path = 'log/' + self.args.dataset + '/' + type(best_model).__name__ + '/' + str(order) + '.mdl'
						torch.save(deepcopy(best_model).to('cpu'), file_path)
					return best_model, best_eval
		except KeyboardInterrupt:
			print('Early stopped manually.')
			print('Training stopped.')
			print('valid:\t', performance_str(best_eval))
			if self.args.save_model:
				file_path = 'log/' + self.args.dataset + '/' + type(best_model).__name__ + '/' + str(order) + '.mdl'				
				torch.save(deepcopy(best_model).to('cpu'), file_path)
			return best_model, best_eval


	

