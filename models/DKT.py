from BaseModel import BaseModel
import torch.nn as nn
import torch

class DKT(BaseModel):

	@staticmethod
	def parse_args(parser):
		super(DKT, DKT).parse_args(parser)
		parser.add_argument('--d_hidden', type = int, default = 128,
			help = 'Dimension # of hidden states.')
		parser.add_argument('--n_layers', type = int, default = 1,
			help = '# of model layers.')

	def __init__(self, args):
		super().__init__(args)
		self.knows_embedding = nn.Embedding(2*args.n_knows + 2, args.d_hidden, padding_idx = 0)
		self.layers = nn.LSTM(
			args.d_hidden,
			args.d_hidden,
			num_layers = args.n_layers,
			batch_first = True,
		)
		self.output_linear = nn.Linear(args.d_hidden, args.n_knows + 1)

	def forward(self, feed_dict):

		self.layers.flatten_parameters()

		knows = feed_dict['knows']		# B, S, K
		corrs = feed_dict['corrs']		# B, S
		probs = feed_dict['probs']		# B, S
		(B, S, K), D = knows.size(), self.args.d_hidden

		bias_knows = (knows > 0)*(knows + (knows > 0)*\
			(corrs.unsqueeze(-1))*(self.args.n_knows + 1))		# B, S, K
		inputs = self.knows_embedding(bias_knows).sum(-2)		# B, S, D

		output, (_, _) = self.layers(inputs)					# B, S, D
		logits = self.output_linear(output)						# B, S, nk

		logits = logits[:, :-1]
		knows = knows[:, 1:]

		scores = logits.gather(-1, knows).sigmoid()				# B, S-1, K
		scores = scores.masked_fill(knows == 0, 0)				# B, S-1, K
		scores = scores.sum(-1)									# B, S-1
		scores = scores / (knows > 0).sum(-1).clamp(1)			# B, S-1
		feed_dict['scores'] = scores[feed_dict['filt']]
		feed_dict['ori_scores'] = scores
	
	