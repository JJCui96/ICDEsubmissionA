from BaseModel import BaseModel
import torch.nn as nn
import numpy as np
import tqdm
import torch

class QTE(BaseModel):

	@staticmethod
	def parse_args(parser):
		super(QTE, QTE).parse_args(parser)
		parser.add_argument('--d_hidden', type = int, default = 128,
			help = 'Dimension # of hidden states.')
		parser.add_argument('--n_freqs', type = int, default = 10,
			help = '# of frequence slots.')
		parser.add_argument('--l2', type = float, default = 0,
			help = 'L2 regularization.')

	def get_optimizer(self):
		return torch.optim.Adam(self.parameters(), 
			lr = self.args.lr, weight_decay = self.args.l2)

	def __init__(self, args):
		super().__init__(args)
		self.knows_embedding = nn.Embedding(args.n_knows + 1, args.d_hidden, padding_idx = 0)
		self.probs_embedding = nn.Embedding(args.n_probs + 1, args.d_hidden, padding_idx = 0)
		self.freq_embedding = nn.Embedding(args.n_freqs + 1, args.d_hidden, padding_idx = 0)
		self.predict = nn.Sequential(
			nn.Linear(args.d_hidden, args.d_hidden//2),
			nn.ReLU(),
			nn.Linear(args.d_hidden//2, 1)
		)
		self.freq_slot = None

	def get_feed_dict(self, batch, mode):
		feed_dict = super().get_feed_dict(batch, mode)
		feed_dict['labels'] = feed_dict['corrs'][feed_dict['probs'] > 0]
		return feed_dict

	def to(self, device):

		if self.freq_slot != None:
			self.freq_slot = self.freq_slot.to(device)
		return super().to(device)

	def forward(self, feed_dict):

		knows = feed_dict['knows']		# B, S, K
		probs = feed_dict['probs']		# B, S
		(B, S, K), D = knows.size(), self.args.d_hidden
		
		knows_emb = self.knows_embedding(knows).sum(-2)
		probs_emb = self.probs_embedding(probs)
		freqs_emb = self.freq_embedding(self.freq_slot[probs]).sigmoid()
		input_emb = knows_emb*freqs_emb + probs_emb*(1 - freqs_emb)
		
		scores = self.predict(input_emb).sigmoid()
		scores = scores.squeeze(-1)

		ori_scores = scores
		scores = scores[feed_dict['probs'] > 0]
		feed_dict['scores'] = scores
		feed_dict['ori_scores'] = ori_scores
	
	def get_emb(self, knows, probs):
		

		knows_emb = self.knows_embedding(knows).sum(-2)
		probs_emb = self.probs_embedding(probs)
		freqs_emb = self.freq_embedding(self.freq_slot[probs]).sigmoid()
		input_emb = knows_emb*freqs_emb + probs_emb*(1 - freqs_emb)
		
		return input_emb
	
	def fit(self, train_set, valid_set, order):	

		freqs = np.zeros(self.args.n_probs + 1)
		
		for seq in tqdm.tqdm(train_set, leave = False, ncols = 50, mininterval = 1):
			for inter in seq:
				prob = inter[0]
				freqs[prob] += 1

		total = np.sum(freqs)
		chunk = total/self.args.n_freqs
		idx = np.argsort(freqs)	
		
		freq_slot = np.zeros(self.args.n_probs + 1)
		s = 0
		k = 1
		for i in idx:
			freq_slot[i] = k
			s += freqs[i]
			if s >= chunk:
				s = 0
				k += 1
		
		self.freq_slot = torch.from_numpy(freq_slot).to(self.device).long()
		return super().fit(train_set, valid_set, order)