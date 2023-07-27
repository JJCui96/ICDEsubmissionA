from BaseModel import BaseModel
import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
from utils import *
from collections import defaultdict

def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
	"""
	This is called by Multi-head atention object to find the values.
	"""
	scores = torch.matmul(q, k.transpose(-2, -1)) / \
		math.sqrt(d_k)  # BS, 8, seqlen, seqlen
	bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

	x1 = torch.arange(seqlen).expand(seqlen, -1).to(q.device)
	x2 = x1.transpose(0, 1).contiguous()

	with torch.no_grad():
		scores_ = scores.masked_fill(mask == 0, -1e32)
		scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
		scores_ = scores_ * mask.float().to(q.device)
		distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
		disttotal_scores = torch.sum(
			scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
		position_effect = torch.abs(
			x1-x2)[None, None, :, :].type(torch.FloatTensor).to(q.device)  # 1, 1, seqlen, seqlen
		# bs, 8, sl, sl positive distance
		dist_scores = torch.clamp(
			(disttotal_scores-distcum_scores)*position_effect, min=0.)
		dist_scores = dist_scores.sqrt().detach()
	m = nn.Softplus()
	gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
	# Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
	total_effect = torch.clamp(torch.clamp(
		(dist_scores*gamma).exp(), min=1e-5), max=1e5)
	scores = scores * total_effect

	scores.masked_fill_(mask == 0, -1e32)
	scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
	if zero_pad:
		pad_zero = torch.zeros(bs, head, 1, seqlen).to(q.device)
		scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
	scores = dropout(scores)
	output = torch.matmul(scores, v)
	return output

class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, d_feature, n_heads, dropout, bias=True):
		super().__init__()
		"""
		It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
		"""
		self.d_model = d_model
		self.d_k = d_feature
		self.h = n_heads

		self.v_linear = nn.Linear(d_model, d_model, bias=bias)
		self.k_linear = nn.Linear(d_model, d_model, bias=bias)
		self.q_linear = nn.Linear(d_model, d_model, bias=bias)
		self.dropout = nn.Dropout(dropout)
		self.proj_bias = bias
		self.out_proj = nn.Linear(d_model, d_model, bias=bias)
		self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
		torch.nn.init.xavier_uniform_(self.gammas)

	def forward(self, q, k, v, mask, zero_pad):

		bs = q.size(0)

		# perform linear operation and split into h heads

		k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
		q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
		v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

		# transpose to get dimensions bs * h * sl * d_model

		k = k.transpose(1, 2)
		q = q.transpose(1, 2)
		v = v.transpose(1, 2)
		# calculate attention using function we will define next
		gammas = self.gammas
		scores = attention(q, k, v, self.d_k,
						   mask, self.dropout, zero_pad, gammas)

		# concatenate heads and put through final linear layer
		concat = scores.transpose(1, 2).contiguous()\
			.view(bs, -1, self.d_model)

		output = self.out_proj(concat)

		return output

class TransformerLayer(nn.Module):
	def __init__(self, d_model, d_feature,
				 d_ff, n_heads, dropout):
		super().__init__()
		"""
			This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
		"""
		# Multi-Head Attention Block
		self.masked_attn_head = MultiHeadAttention(
			d_model, d_feature, n_heads, dropout)

		# Two layer norm layer and two droput layer
		self.layer_norm1 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)

		self.linear1 = nn.Linear(d_model, d_ff)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(d_ff, d_model)

		self.layer_norm2 = nn.LayerNorm(d_model)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, mask, query, key, values, apply_pos=True):

		seqlen, batch_size = query.size(1), query.size(0)
		nopeek_mask = np.triu(
			np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
		src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)
		if mask == 0:  # If 0, zero-padding is needed.
			# Calls block.masked_attn_head.forward() method
			query2 = self.masked_attn_head(
				query, key, values, mask=src_mask, zero_pad=True)
		else:
			# Calls block.masked_attn_head.forward() method
			query2 = self.masked_attn_head(
				query, key, values, mask=src_mask, zero_pad=False)

		query = query + self.dropout1((query2))
		query = self.layer_norm1(query)
		if apply_pos:
			query2 = self.linear2(self.dropout(
				self.activation(self.linear1(query))))
			query = query + self.dropout2((query2))
			query = self.layer_norm2(query)
		return query

class Architecture(nn.Module):
	def __init__(self,  n_blocks, d_model,
				 d_ff, n_heads, dropout):
		super().__init__()
		"""
			n_block : number of stacked blocks in the attention
			d_model : dimension of attention input/output
			d_feature : dimension of input in each of the multi-head attention part.
			n_head : number of heads. n_heads*d_feature = d_model
		"""
		self.d_model = d_model

		self.blocks_1 = nn.ModuleList([
			TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
								d_ff=d_ff, dropout=dropout, n_heads=n_heads)
			for _ in range(n_blocks)
		])
		self.blocks_2 = nn.ModuleList([
			TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
								d_ff=d_ff, dropout=dropout, n_heads=n_heads)
			for _ in range(n_blocks*2)
		])

	def forward(self, q_embed_data, qa_embed_data):
		# target shape  bs, seqlen
		seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

		qa_pos_embed = qa_embed_data
		q_pos_embed = q_embed_data

		y = qa_pos_embed
		seqlen, batch_size = y.size(1), y.size(0)
		x = q_pos_embed

		# encoder
		for block in self.blocks_1:  # encode qas
			y = block(mask=1, query=y, key=y, values=y)
		flag_first = True
		for block in self.blocks_2:
			if flag_first:  # peek current question
				x = block(mask=1, query=x, key=x,
						  values=x, apply_pos=False)
				flag_first = False
			else:  # dont peek current response
				x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
				flag_first = True
		return x



class AKT(BaseModel):

	@staticmethod
	def parse_args(parser):
		super(AKT, AKT).parse_args(parser)
		parser.add_argument('--d_hidden', type = int, default = 128,
			help = 'Dimension # of hidden states.')
		parser.add_argument('--n_layers', type = int, default = 2,
			help = '# of model layers.')
		parser.add_argument('--dropout', type = float, default = 0.0,
			help = 'Dropout in rnn.')
		parser.add_argument('--pl2', type = float, default = 0.0,
			help = 'L2 coefficient of question embedding.')
		parser.add_argument('--n_heads', type = int, default = 4,
			help = '# of attention heads.')


	def __init__(self, args):
		super(AKT, self).__init__(args)
		self.n_question = args.n_knows
		self.dropout = args.dropout
		self.n_pid = args.n_probs
		self.l2 = args.pl2
		embed_l = args.d_hidden
		self.difficult_param = nn.Embedding(self.n_pid+1, 1, padding_idx = 0)
		self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l, padding_idx = 0)
		self.qa_embed_diff = nn.Embedding(2 * self.n_question + 2, embed_l, padding_idx = 0)
		self.q_embed = nn.Embedding(self.n_question+1, embed_l, padding_idx = 0)
		self.qa_embed = nn.Embedding(2*self.n_question+2, embed_l, padding_idx = 0)

		self.model = Architecture(
			n_blocks=args.n_layers, n_heads=args.n_heads, dropout=self.dropout,
			d_model=embed_l, d_ff=embed_l)

		self.out = nn.Sequential(
			nn.Linear(2*embed_l, embed_l),
			nn.ReLU(),
			nn.Dropout(self.dropout),
			nn.Linear(embed_l, embed_l//2),
			nn.ReLU(),
			nn.Dropout(self.dropout),
			nn.Linear(embed_l//2, 1)
		)

	def loss(self, feed_dict):
		ori_loss = super(AKT, self).loss(feed_dict)
		reg_loss = feed_dict['pl2_loss']
		return ori_loss + reg_loss


	def forward(self, feed_dict):

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

		concat_q = torch.cat([d_output, q_embed_data], dim=-1)
		scores = self.out(concat_q).sigmoid().squeeze(-1)
		ori_scores = scores
		feed_dict['pl2_loss'] = c_reg_loss

		feed_dict['scores'] = scores[:, 1:][feed_dict['filt']]
		feed_dict['ori_scores'] = ori_scores[:, 1:]
