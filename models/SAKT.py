from BaseModel import BaseModel
import torch.nn as nn
import torch
from utils import *

class SAKT(BaseModel):

	@staticmethod
	def parse_args(parser):
		super(SAKT, SAKT).parse_args(parser)
		parser.add_argument('--d_hidden', type = int, default = 128,
			help = 'Dimension # of hidden states.')
		parser.add_argument('--l2', type = float, default = 0,
			help = 'L2 regularization.')
		parser.add_argument('--dropout', type = float, default = 0.0,
			help = 'Dropout ratio.')
		parser.add_argument('--n_heads', type = int, default = 4,
			help = '# of attention heads.')

	def get_optimizer(self):
		return torch.optim.Adam(self.parameters(), 
			lr = self.args.lr, weight_decay = self.args.l2)

	def __init__(self, args):
		super(SAKT, self).__init__(args)
		self.inter_embedding = nn.Embedding(2*args.n_knows + 2, args.d_hidden, padding_idx = 0)
		self.knows_embedding = nn.Embedding(args.n_knows + 1, args.d_hidden, padding_idx = 0)
		self.pos_embedding = nn.Embedding(101, args.d_hidden, padding_idx = 0)
		self.attn = nn.MultiheadAttention(
			embed_dim = self.args.d_hidden,
			num_heads = self.args.n_heads,
			dropout = self.args.dropout,
			batch_first = True,
		)
		self.layer_norm1 = nn.LayerNorm(self.args.d_hidden)
		self.layer_norm2 = nn.LayerNorm(self.args.d_hidden)
		self.FFN = nn.Sequential(
			nn.Linear(self.args.d_hidden, self.args.d_hidden),
			nn.ReLU(),
			nn.Linear(self.args.d_hidden, self.args.d_hidden),
			nn.Dropout(self.args.dropout)
		)
		self.output_linear = nn.Linear(self.args.d_hidden, 1)

	def forward(self, feed_dict):

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
		scores = self.output_linear(output).sigmoid().squeeze(-1)# B, S

		feed_dict['scores'] = scores[feed_dict['filt']]
		feed_dict['ori_scores'] = scores