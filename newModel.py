# -*- coding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.init import xavier_normal_, constant_
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, GCNConv
from torch.nn import Module
import torch.nn.functional as F
import numpy as np


class GlobalGNN(Module):
    def __init__(self, args):
        super(GlobalGNN, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        in_channels = hidden_channels = self.hidden_size
        self.num_layers = len(args.sample_size)
        self.dropout = nn.Dropout(args.gnn_dropout_prob)
        self.gcn = GCNConv(self.hidden_size, self.hidden_size)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))
        for i in range(self.num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))

    def forward(self, x, adjs, attr):
        xs = []
        x_all = x
        if self.num_layers > 1:
            # 编码策略： GCN with edge weight + GraphSage
            # for i, (edge_index, e_id, size) in enumerate(adjs):
            #     weight = attr[e_id].view(-1).type(torch.float)
            #
            #     x = x_all
            #     if len(list(x.shape)) < 2:
            #         x = x.unsqueeze(0)
            #     # base GCN include inloop, single direaction
            #     x = self.gcn(x, edge_index, weight)
            #     # sage
            #     x_target = x[:size[1]]  # Target nodes are always placed first.
            #     x = self.convs[i]((x, x_target), edge_index)
            #     if i != self.num_layers - 1:
            #         x = F.relu(x)
            #         x = self.dropout(x)

            # 编码策略：GCN with edge weight
            for i, (edge_index, e_id, size) in enumerate(adjs):
                weight = attr[e_id].view(-1).type(torch.float)
                if len(list(x.shape)) < 2:
                    x = x.unsqueeze(0)
                x = self.gcn(x, edge_index, weight)
                x = x[:size[1]]
                # # sage
                # x_target = x[:size[1]]  # Target nodes are always placed first.
                # x = self.convs[i]((x, x_target), edge_index)
                # if i != self.num_layers - 1:
                #     x = F.relu(x)
                #     x = self.dropout(x)
        else:
            # 只有1-hop的情況
            edge_index, e_id, size = adjs.edge_index, adjs.e_id, adjs.size
            x = x_all
            x = self.dropout(x)
            weight = attr[e_id].view(-1).type(torch.float)
            if len(list(x.shape)) < 2:
                x = x.unsqueeze(0)
            x = self.gcn(x, edge_index, weight)
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[-1]((x, x_target), edge_index)
        xs.append(x)
        return torch.cat(xs, 0)


class GCL4SR(nn.Module):
    def __init__(self, args, global_graph):
        super(GCL4SR, self).__init__()
        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda:{}".format(self.args.gpu_id) if self.cuda_condition else "cpu")
        self.global_graph = global_graph.to(self.device)
        self.global_gnn = GlobalGNN(args)

        self.cluster_k = 5
        self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        # sequence encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size,
                                                        nhead=args.num_attention_heads,
                                                        dim_feedforward=4 * args.hidden_size,
                                                        dropout=args.attention_probs_dropout_prob,
                                                        activation=args.hidden_act)
        self.item_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.num_hidden_layers)

        # InterestAttention
        self.short_user_embeddings = nn.Embedding(args.user_size, args.hidden_size)
        self.LongTermMHA = nn.MultiheadAttention(embed_dim=args.hidden_size, num_heads=1, dropout=0.1)
        self.ShortTermMHA = nn.MultiheadAttention(embed_dim=args.hidden_size, num_heads=1, dropout=0.1)
        self.GRULayer = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=1)
        self.SeqLayer = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=1)
        self.GRULayer1 = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=1)
        self.linear_LSTerm1 = nn.Linear(args.hidden_size * 3, args.hidden_size)
        self.linear_LSTerm2 = nn.Linear(args.hidden_size, 1)

        # Assigment Matrix
        self.w_c1 = nn.Parameter(torch.Tensor(args.hidden_size, args.hidden_size))

        self.w_c2 = nn.Parameter(torch.Tensor(args.hidden_size, self.cluster_k))


        # AttNet
        self.w_1 = nn.Parameter(torch.Tensor(2 * args.hidden_size, args.hidden_size))
        self.w_2 = nn.Parameter(torch.Tensor(args.hidden_size, 1))
        self.linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_2 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        # fast run with mmd
        # self.w_g = nn.Linear(args.hidden_size, 1)
        # self.w_e = nn.Linear(args.hidden_size, 1)

        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.linear_transform = nn.Linear(3 * args.hidden_size, args.hidden_size, bias=False)
        self.gnndrop = nn.Dropout(args.gnn_dropout_prob)

        self.criterion = nn.CrossEntropyLoss()
        self.betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=self.betas,
                                          weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.args = args
        self.apply(self._init_weights)

        # user-specific gating
        self.gate_item = Variable(torch.zeros(args.hidden_size, 1).type
                                  (torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_user = Variable(torch.zeros(args.hidden_size, args.max_seq_length).type
                                  (torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_item = torch.nn.init.xavier_uniform_(self.gate_item)
        self.gate_user = torch.nn.init.xavier_uniform_(self.gate_user)

    def _init_weights(self, module):
        """ Initialize the weights """
        stdv = 1.0 / math.sqrt(self.args.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def MMD_loss(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        if self.args.fast_run:
            source = source.view(-1, self.args.max_seq_length)
            target = target.view(-1, self.args.max_seq_length)
            batch_size = int(source.size()[0])
            loss_all = []
            kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                           fix_sigma=fix_sigma)
            xx = kernels[:batch_size, :batch_size]
            yy = kernels[batch_size:, batch_size:]
            xy = kernels[:batch_size, batch_size:]
            yx = kernels[batch_size:, :batch_size]
            loss = torch.mean(xx + yy - xy - yx)
            loss_all.append(loss)
        else:
            source = source.view(-1, self.args.max_seq_length, self.args.hidden_size)
            target = target.view(-1, self.args.max_seq_length, self.args.hidden_size)
            batch_size = int(source.size()[1])
            loss_all = []
            for i in range(int(source.size()[0])):
                kernels = self.guassian_kernel(source[i], target[i], kernel_mul=kernel_mul, kernel_num=kernel_num,
                                               fix_sigma=fix_sigma)
                xx = kernels[:batch_size, :batch_size]
                yy = kernels[batch_size:, batch_size:]
                xy = kernels[:batch_size, batch_size:]
                yx = kernels[batch_size:, :batch_size]
                loss = torch.mean(xx + yy - xy - yx)
                loss_all.append(loss)
        return sum(loss_all) / len(loss_all)

    def GCL_loss(self, hidden, hidden_norm=True, temperature=1.0):
        batch_size = hidden.shape[0] // 2
        LARGE_NUM = 1e9
        # inner dot or cosine
        if hidden_norm:
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        hidden_list = torch.split(hidden, batch_size, dim=0)
        hidden1, hidden2 = hidden_list[0], hidden_list[1]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.from_numpy(np.arange(batch_size)).to(hidden.device)
        masks = torch.nn.functional.one_hot(torch.from_numpy(np.arange(batch_size)).to(hidden.device), batch_size)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(1, 0)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(1, 0)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(1, 0)) / temperature
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(1, 0)) / temperature

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
        loss = (loss_a + loss_b)
        return loss

    def gnn_encode(self, items):
        subgraph_loaders = NeighborSampler(self.global_graph.edge_index, node_idx=items, sizes=self.args.sample_size,
                                           shuffle=False,
                                           num_workers=0, batch_size=items.shape[0])
        g_adjs = []
        s_nodes = []
        for (b_size, node_idx, adjs) in subgraph_loaders:
            if type(adjs) == list:
                g_adjs = [adj.to(items.device) for adj in adjs]
            else:
                g_adjs = adjs.to(items.device)
            n_idxs = node_idx.to(items.device)
            s_nodes = self.item_embeddings(n_idxs).squeeze()
        attr = self.global_graph.edge_attr.to(items.device)
        g_hidden = self.global_gnn(s_nodes, g_adjs, attr)
        return g_hidden

    def final_att_net(self, seq_mask, hidden):
        batch_size = hidden.shape[0]
        lens = hidden.shape[1]
        pos_emb = self.position_embeddings.weight[:lens]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        seq_hidden = torch.sum(hidden * seq_mask, -2) / torch.sum(seq_mask, 1)
        seq_hidden = seq_hidden.unsqueeze(-2).repeat(1, lens, 1)
        item_hidden = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        item_hidden = torch.tanh(item_hidden)
        score = torch.sigmoid(self.linear_1(item_hidden) + self.linear_2(seq_hidden))
        att_score = torch.matmul(score, self.w_2)
        att_score_masked = att_score * seq_mask
        output = torch.sum(att_score_masked * hidden, 1)
        return output

    def Interest_net(self, user_embedding, action_embedding, target_embedding, seq_mask, k=3):
        '''
        :param user_embedding: shape(batch_size, dim)
        :param action_embedding: shape(batch_size, seq_len, dim)
        :param target_embedding: shape(batch_size, dim)
        :param seq_mask: shape(batch_size, seq_len)  =1 padding  =0 real data
        :param k: short interest's sequence length
        :return: shape(batch_size, dim)
        '''

        # calculate user's LS-Term Interest
        user_embedding = user_embedding.unsqueeze(0)
        u_l = self.LongTermMHA(user_embedding, action_embedding.permute(1, 0, 2), action_embedding.permute(1, 0, 2),
                               key_padding_mask=seq_mask, need_weights=False)[0].permute(1, 0, 2).squeeze()
        _, h_out = self.GRULayer(action_embedding.permute(1, 0, 2), user_embedding)
        short_embedding, _ = self.SeqLayer(action_embedding.permute(1, 0, 2))
        u_s = self.ShortTermMHA(h_out, short_embedding, short_embedding, key_padding_mask=seq_mask,
                                need_weights=False)[0].permute(1, 0, 2).squeeze()
        user_embedding = user_embedding.squeeze()

        # design proxy label for LS-Term Interest representation
        proxy_mask = (1. - seq_mask.float()).unsqueeze(-1)
        p_l = torch.sum(action_embedding * proxy_mask, dim=1) / torch.sum(proxy_mask, dim=1)

        position_mask = (torch.arange(0, proxy_mask.shape[1]) > (proxy_mask.shape[1] - k - 1)).float().unsqueeze(-1).to(proxy_mask.device)
        short_proxy_mask = torch.logical_and(position_mask, proxy_mask).float()
        p_s = torch.sum(action_embedding * short_proxy_mask, dim=1) / torch.sum(short_proxy_mask, dim=1)

        # # Adaptive Fusion for Interaction Prediction
        _, final_state = self.GRULayer1(action_embedding.permute(1, 0, 2))
        # concat_all = torch.cat([final_state.squeeze(), target_embedding, u_l, u_s], dim=-1)
        concat_all = torch.cat([final_state.squeeze(), u_l, u_s], dim=-1)
        alpha = torch.sigmoid(self.linear_LSTerm2(torch.sigmoid(self.linear_LSTerm1(concat_all))))
        final_user_embed = alpha * u_l + (1 - alpha) * u_s


        return final_user_embed, u_l, u_s, p_l, p_s

    def Interest_Predict(self, user_embed, targets):
        '''
        :param user_embed: shape(batch_size, dim)
        :param target_embed: shape(batch_size,) target item ID
        :return:
        '''
        user_embed = self.dropout(user_embed)
        test_item_emb = self.item_embeddings.weight[:self.args.item_size]
        logits = torch.matmul(user_embed, test_item_emb.transpose(0, 1))
        main_loss = self.criterion(logits, targets)
        return main_loss

    def LSTerm_loss(self, u_l, u_s, p_l, p_s, seq_mask, threshold=3):
        '''
        :param u_l: user long term interest | shape(batch_size, dim)
        :param u_s: user short term interst | shape(batch_size, dim)
        :param p_l: user long interest proxy | shape(batch_size, dim)
        :param p_s: user short interst proxy | shape(batch_size, dim)
        :param seq_mask: user sequence padding mask | shape(batch_size, seq_len)
        :param threshold: consider ls-term interst's seq length
        :return: mean_loss
        '''

        seq_len = torch.sum(seq_mask, dim=-1)
        LSTerm_mask = torch.where(torch.greater(seq_len, threshold), torch.ones_like(seq_len), torch.zeros_like(seq_len))
        long_mean_recent_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(u_l * (-p_l + p_s), dim=-1))) / torch.sum(seq_len, dim=-1)
        short_recent_mean_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(u_s * (-p_s + p_l), dim=-1))) / torch.sum(seq_len, dim=-1)
        mean_long_short_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(p_l * (-u_l + u_s), dim=-1))) / torch.sum(seq_len, dim=-1)
        recent_short_long_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(p_s * (-u_s+ u_l), dim=-1))) / torch.sum(seq_len, dim=-1)
        lsTerm_loss = long_mean_recent_loss + short_recent_mean_loss + mean_long_short_loss + recent_short_long_loss
        return lsTerm_loss



    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').            Unmasked positions are filled with float(0.0).        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -10000.0).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, data):
        """
        :param data: (3, #batch_size, ?) contain 3 tensor,
               data[0] = user_id, data[1] = action_seq(input) data[2] = target item
        :return:
        """

        user_ids = data[0]
        inputs = data[1]

        seq = inputs.flatten()
        seq_mask = (inputs == 0).float().unsqueeze(-1)
        seq_mask = 1.0 - seq_mask

        seq_hidden_global_a = self.gnn_encode(seq).view(-1, self.args.max_seq_length, self.args.hidden_size)
        seq_hidden_global_b = self.gnn_encode(seq).view(-1, self.args.max_seq_length, self.args.hidden_size)

        key_padding_mask = (inputs == 0)
        attn_mask = self.generate_square_subsequent_mask(self.args.max_seq_length).to(inputs.device)
        seq_hidden_local = self.item_embeddings(inputs)
        seq_hidden_local = self.LayerNorm(seq_hidden_local)
        seq_hidden_local = self.dropout(seq_hidden_local)
        seq_hidden_permute = seq_hidden_local.permute(1, 0, 2)

        encoded_layers = self.item_encoder(seq_hidden_permute,
                                           mask=attn_mask,
                                           src_key_padding_mask=key_padding_mask)

        sequence_output = encoded_layers.permute(1, 0, 2)

        user_emb = self.user_embeddings(user_ids).view(-1, self.args.hidden_size)

        gating_score_a = torch.sigmoid(torch.matmul(seq_hidden_global_a, self.gate_item.unsqueeze(0)).squeeze() +
                                       user_emb.mm(self.gate_user))
        user_seq_a = seq_hidden_global_a * gating_score_a.unsqueeze(2)
        gating_score_b = torch.sigmoid(torch.matmul(seq_hidden_global_b, self.gate_item.unsqueeze(0)).squeeze() +
                                       user_emb.mm(self.gate_user))
        user_seq_b = seq_hidden_global_b * gating_score_b.unsqueeze(2)

        user_seq_a = self.gnndrop(user_seq_a)
        user_seq_b = self.gnndrop(user_seq_b)

        hidden = torch.cat([sequence_output, user_seq_a, user_seq_b], -1)
        hidden = self.linear_transform(hidden)

        return sequence_output, hidden, user_seq_a, user_seq_b, (seq_hidden_global_a, seq_hidden_global_b), seq_mask

    def train_stage(self, data):
        targets = data[2]
        sequence_output, hidden, user_seq_a, user_seq_b, (seq_gnn_a, seq_gnn_b), seq_mask = self.forward(data)
        seq_out = self.final_att_net(seq_mask, hidden)
        seq_out = self.dropout(seq_out)
        test_item_emb = self.item_embeddings.weight[:self.args.item_size]
        logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        main_loss = self.criterion(logits, targets)

        sum_a = torch.sum(seq_gnn_a * seq_mask, 1) / torch.sum(seq_mask.float(), 1)
        sum_b = torch.sum(seq_gnn_b * seq_mask, 1) / torch.sum(seq_mask.float(), 1)

        info_hidden = torch.cat([sum_a, sum_b], 0)
        gcl_loss = self.GCL_loss(info_hidden, hidden_norm=True, temperature=0.5)

        # [B, L, d] to [B, L]️, can reduce training time and memory
        if self.args.fast_run:
            seq_hidden_local = self.w_e(self.item_embeddings(data[1])).squeeze().unsqueeze(0)
            user_seq_a = self.w_g(user_seq_a).squeeze()
            user_seq_b = self.w_g(user_seq_b).squeeze()
        else:
            seq_hidden_local = self.item_embeddings(data[1])
        mmd_loss = self.MMD_loss(seq_hidden_local, user_seq_a) + self.MMD_loss(seq_hidden_local, user_seq_b)

        joint_loss = main_loss + self.args.lam1 * gcl_loss + self.args.lam2 * mmd_loss

        return joint_loss, main_loss, gcl_loss, mmd_loss



    def eval_stage(self, data):
        _, hidden, _, _, _, seq_mask = self.forward(data)
        hidden = self.final_att_net(seq_mask, hidden)

        return hidden

    def new_train_stage(self, data):
        user_ids, action_seq, targets = data[0], data[1], data[2]
        user_embed = self.user_embeddings(user_ids)
        seq_embed = self.item_embeddings(action_seq)
        target_embed = self.item_embeddings(targets)
        seq_mask = (action_seq == 0)
        final_user_embed, u_l, u_s, p_l, p_s = self.Interest_net(user_embed, seq_embed, target_embed, seq_mask)
        main_loss = self.Interest_Predict(final_user_embed, targets)
        lsTerm_loss = self.LSTerm_loss(u_l, u_s, p_l, p_s, 1.-seq_mask.float())
        re_loss = torch.FloatTensor([0.0])
        joint_loss = main_loss + lsTerm_loss + 0.
        return joint_loss, main_loss, lsTerm_loss, re_loss

    def new_eval_stage(self, data):
        user_ids, action_seq, targets = data[0], data[1], data[2]
        user_embed = self.user_embeddings(user_ids)
        seq_embed = self.item_embeddings(action_seq)
        target_embed = self.item_embeddings(targets)
        seq_mask = (action_seq == 0)
        final_user_embed, _, _, _, _ = self.Interest_net(user_embed, seq_embed, target_embed, seq_mask)
        return final_user_embed

    # def create_assig_matrix(self, action_embed):


class GRU4Rec(nn.Module):
    def __init__(self, args, global_graph):
        super(GRU4Rec, self).__init__()
        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda:{}".format(self.args.gpu_id) if self.cuda_condition else "cpu")
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.onehot_buffer = self.init_emb()
        self.num_layers = 1
        self.GRULayer = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=self.num_layers, dropout=0.5)
        self.criterion = nn.CrossEntropyLoss()
        self.betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=self.betas,
                                          weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    def forward(self, seq_embed):
        h0 = torch.zeros(self.num_layers, seq_embed.shape[1], self.args.hidden_size).to(self.device)
        seq_embed, hidden = self.GRULayer(seq_embed, h0)
        return hidden

    def init_emb(self):
        '''
        Initialize the one_hot embedding buffer, which will be used for producing the one-hot embeddings efficiently
        '''
        onehot_buffer = torch.FloatTensor(self.args.batch_size, self.args.item_size)
        onehot_buffer = onehot_buffer.to(self.device)
        return onehot_buffer

    def onehot_encode(self, input):
        """
        Returns a one-hot vector corresponding to the input
        Args:
            input (B,): torch.LongTensor of item indices
            buffer (B,output_size): buffer that stores the one-hot vector
        Returns:
            one_hot (B,C): torch.FloatTensor of one-hot vectors
        """
        self.onehot_buffer.zero_()
        index = input.view(-1, 1)
        one_hot = self.onehot_buffer.scatter_(1, index, 1)
        return one_hot

    def init_hidden(self):
        '''
        Initialize the hidden state of the GRU
        '''
        try:
            h0 = torch.zeros(self.num_layers, self.args.batch_size, self.args.hidden_size).to(self.device)
        except:
            self.device = 'cpu'
            h0 = torch.zeros(self.num_layers, self.args.batch_size, self.args.hidden_size).to(self.device)
        return h0

    def train_stage(self, data):
        seq_embed = self.item_embeddings(data[1]).permute(1, 0, 2)
        seq_out = self.forward(seq_embed).squeeze()
        test_item_emb = self.item_embeddings.weight[:self.args.item_size]
        logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        main_loss = self.criterion(logits, data[2])
        reg_loss = torch.FloatTensor([0.])
        aaa_loss = torch.FloatTensor([0.])
        joint_loss = main_loss + 0. + 0.
        return joint_loss, main_loss, reg_loss, aaa_loss

    def eval_stage(self, data):
        seq_embed = self.item_embeddings(data[1]).permute(1, 0, 2)
        seq_out = self.forward(seq_embed).squeeze()
        return seq_out

class LS4Rec(nn.Module):
    def __init__(self, args, global_graph):
        super(LS4Rec, self).__init__()

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda:{}".format(self.args.gpu_id) if self.cuda_condition else "cpu")
        self.global_graph = global_graph.to(self.device)
        self.global_gnn = GlobalGNN(args)

        self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        # sequence encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size,
                                                        nhead=args.num_attention_heads,
                                                        dim_feedforward=4 * args.hidden_size,
                                                        dropout=args.attention_probs_dropout_prob,
                                                        activation=args.hidden_act)
        self.item_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.num_hidden_layers)

        # InterestAtten
        self.short_user_embeddings = nn.Embedding(args.user_size, args.hidden_size)
        self.long_user_embeddings = nn.Embedding(args.user_size, args.hidden_size)
        self.LongTermMHA = nn.MultiheadAttention(embed_dim=args.hidden_size,
                                                 num_heads=args.num_attention_heads,
                                                 dropout=0.0)

        self.ShortTermMHA = nn.MultiheadAttention(embed_dim=args.hidden_size,
                                                  num_heads=args.num_attention_heads,
                                                  dropout=0.0)
        self.GRULayer = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=1)
        self.SeqLayer = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=1)
        self.GRULayer1 = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=1)
        self.linear_LSTerm1 = nn.Linear(args.hidden_size * 3, args.hidden_size)
        self.linear_LSTerm2 = nn.Linear(args.hidden_size, 1)

        self.mse = nn.MSELoss()

        # AttNet
        self.w_1 = nn.Parameter(torch.Tensor(2 * args.hidden_size, args.hidden_size))
        self.w_2 = nn.Parameter(torch.Tensor(args.hidden_size, 1))
        self.linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_2 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        # fast run with mmd
        self.w_g = nn.Linear(args.hidden_size, 1)
        self.w_e = nn.Linear(args.hidden_size, 1)

        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.linear_transform = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        self.gnndrop = nn.Dropout(args.gnn_dropout_prob)

        self.criterion = nn.CrossEntropyLoss()
        self.betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=self.betas,
                                          weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.args = args
        self.apply(self._init_weights)

        # user-specific gating
        self.gate_item = Variable(torch.zeros(args.hidden_size, 1).type
                                  (torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_user = Variable(torch.zeros(args.hidden_size, args.max_seq_length).type
                                  (torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_item = torch.nn.init.xavier_uniform_(self.gate_item)
        self.gate_user = torch.nn.init.xavier_uniform_(self.gate_user)

    def _init_weights(self, module):
        """ Initialize the weights """
        stdv = 1.0 / math.sqrt(self.args.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def MMD_loss(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        if self.args.fast_run:
            source = source.view(-1, self.args.max_seq_length)
            target = target.view(-1, self.args.max_seq_length)
            batch_size = int(source.size()[0])
            loss_all = []
            kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                           fix_sigma=fix_sigma)
            xx = kernels[:batch_size, :batch_size]
            yy = kernels[batch_size:, batch_size:]
            xy = kernels[:batch_size, batch_size:]
            yx = kernels[batch_size:, :batch_size]
            loss = torch.mean(xx + yy - xy - yx)
            loss_all.append(loss)
        else:
            source = source.view(-1, self.args.max_seq_length, self.args.hidden_size)
            target = target.view(-1, self.args.max_seq_length, self.args.hidden_size)
            batch_size = int(source.size()[1])
            loss_all = []
            for i in range(int(source.size()[0])):
                kernels = self.guassian_kernel(source[i], target[i], kernel_mul=kernel_mul, kernel_num=kernel_num,
                                               fix_sigma=fix_sigma)
                xx = kernels[:batch_size, :batch_size]
                yy = kernels[batch_size:, batch_size:]
                xy = kernels[:batch_size, batch_size:]
                yx = kernels[batch_size:, :batch_size]
                loss = torch.mean(xx + yy - xy - yx)
                loss_all.append(loss)
        return sum(loss_all) / len(loss_all)

    def GCL_loss(self, hidden, hidden_norm=True, temperature=1.0):
        batch_size = hidden.shape[0] // 2
        LARGE_NUM = 1e9
        # inner dot or cosine
        if hidden_norm:
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        hidden_list = torch.split(hidden, batch_size, dim=0)
        hidden1, hidden2 = hidden_list[0], hidden_list[1]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.from_numpy(np.arange(batch_size)).to(hidden.device)
        masks = torch.nn.functional.one_hot(torch.from_numpy(np.arange(batch_size)).to(hidden.device), batch_size)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(1, 0)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(1, 0)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(1, 0)) / temperature
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(1, 0)) / temperature

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
        loss = (loss_a + loss_b)
        return loss

    def LSTerm_loss(self, u_l, u_s, p_l, p_s, seq_mask, threshold=3, margin=1.0):
        '''
        :param u_l: user long term interest | shape(batch_size, dim)
        :param u_s: user short term interst | shape(batch_size, dim)
        :param p_l: user long interest proxy | shape(batch_size, dim)
        :param p_s: user short interst proxy | shape(batch_size, dim)
        :param seq_mask: user sequence padding mask | shape(batch_size, seq_len)
        :param threshold: consider ls-term interst's seq length
        :return: mean_loss
        '''

        seq_len = torch.sum(seq_mask, dim=-1)
        LSTerm_mask = torch.where(torch.greater(seq_len, threshold), torch.ones_like(seq_len), torch.zeros_like(seq_len))
        if self.args.lsTerm_loss == 'bpr':
            long_mean_recent_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(u_l * (-p_l + p_s), dim=-1))) / torch.sum(LSTerm_mask)
            short_recent_mean_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(u_s * (-p_s + p_l), dim=-1))) / torch.sum(LSTerm_mask)
            mean_long_short_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(p_l * (-u_l + u_s), dim=-1))) / torch.sum(LSTerm_mask)
            recent_short_long_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(p_s * (-u_s+ u_l), dim=-1))) / torch.sum(LSTerm_mask)
        elif self.args.lsTerm_loss == 'triplet':
            distance_long_mean = torch.square_(u_l - p_l)
            distance_long_recent = torch.square_(u_l - p_s)
            distance_short_mean = torch.square_(u_s - p_l)
            distance_short_recent = torch.square_(u_s - p_s)
            long_mean_recent_loss = torch.sum(
                LSTerm_mask * torch.sum(torch.maximum(0.0, distance_long_mean - distance_long_recent + margin),
                                        dim=-1)) / torch.sum(LSTerm_mask)
            short_recent_mean_loss = torch.sum(
                LSTerm_mask * torch.sum(torch.maximum(0.0, distance_short_recent - distance_short_mean + margin),
                                        dim=-1)) / torch.sum(LSTerm_mask)
            mean_long_short_loss = torch.sum(
                LSTerm_mask * torch.sum(torch.maximum(0.0, distance_long_mean - distance_short_mean + margin),
                                        dim=-1)) / torch.sum(LSTerm_mask)
            recent_short_long_loss = torch.sum(
                LSTerm_mask * torch.sum(torch.maximum(0.0, distance_short_recent - distance_long_recent + margin),
                                        dim=-1)) / torch.sum(LSTerm_mask)
        lsTerm_loss = long_mean_recent_loss + short_recent_mean_loss + mean_long_short_loss + recent_short_long_loss
        return lsTerm_loss

    def gnn_encode(self, items):
        subgraph_loaders = NeighborSampler(self.global_graph.edge_index, node_idx=items, sizes=self.args.sample_size,
                                           shuffle=False,
                                           num_workers=0, batch_size=items.shape[0])
        g_adjs = []
        s_nodes = []
        for (b_size, node_idx, adjs) in subgraph_loaders:
            if type(adjs) == list:
                g_adjs = [adj.to(items.device) for adj in adjs]
            else:
                g_adjs = adjs.to(items.device)
            n_idxs = node_idx.to(items.device)
            s_nodes = self.item_embeddings(n_idxs).squeeze()
        attr = self.global_graph.edge_attr.to(items.device)
        g_hidden = self.global_gnn(s_nodes, g_adjs, attr)
        return g_hidden

    def final_att_net(self, seq_mask, hidden):
        batch_size = hidden.shape[0]
        lens = hidden.shape[1]
        pos_emb = self.position_embeddings.weight[:lens]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        # print(seq_mask.shape, hidden.shape)
        seq_hidden = torch.sum(hidden * seq_mask, -2) / torch.sum(seq_mask, 1)
        seq_hidden_r = seq_hidden.unsqueeze(-2).repeat(1, lens, 1)
        item_hidden = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        item_hidden = torch.tanh(item_hidden)
        score = torch.sigmoid(self.linear_1(item_hidden) + self.linear_2(seq_hidden_r))
        att_score = torch.matmul(score, self.w_2)
        att_score_masked = att_score * seq_mask
        output = torch.sum(att_score_masked * hidden, 1)
        return output, seq_hidden


    def Interest_net(self, user_embedding, action_embedding, target_embedding, seq_mask, k=3):
        '''
        :param user_embedding: shape(batch_size, dim)
        :param action_embedding: shape(batch_size, seq_len, dim)
        :param target_embedding: shape(batch_size, dim)
        :param seq_mask: shape(batch_size, seq_len)  =1 padding  =0 real data
        :param k: short interest's sequence length
        :return: shape(batch_size, dim)
        '''

        # calculate user's LS-Term Interest
        user_embedding = user_embedding.unsqueeze(0)
        u_l = self.LongTermMHA(user_embedding, action_embedding.permute(1, 0, 2), action_embedding.permute(1, 0, 2),
                               key_padding_mask=seq_mask, need_weights=False)[0].permute(1, 0, 2).squeeze()
        _, h_out = self.GRULayer(action_embedding.permute(1, 0, 2), user_embedding)
        short_embedding, _ = self.SeqLayer(action_embedding.permute(1, 0, 2))
        u_s = self.ShortTermMHA(h_out, short_embedding, short_embedding, key_padding_mask=seq_mask,
                                need_weights=False)[0].permute(1, 0, 2).squeeze()
        user_embedding = user_embedding.squeeze()

        # design proxy label for LS-Term Interest representation
        proxy_mask = (1. - seq_mask.float()).unsqueeze(-1)
        p_l = torch.sum(action_embedding * proxy_mask, dim=1) / torch.sum(proxy_mask, dim=1)

        position_mask = (torch.arange(0, proxy_mask.shape[1]) > (proxy_mask.shape[1] - k - 1)).float().unsqueeze(-1).to(proxy_mask.device)
        short_proxy_mask = torch.logical_and(position_mask, proxy_mask).float()
        p_s = torch.sum(action_embedding * short_proxy_mask, dim=1) / torch.sum(short_proxy_mask, dim=1)

        # # Adaptive Fusion for Interaction Prediction
        _, final_state = self.GRULayer1(action_embedding.permute(1, 0, 2))
        # concat_all = torch.cat([final_state.squeeze(), target_embedding, u_l, u_s], dim=-1)
        concat_all = torch.cat([final_state.squeeze(), u_l, u_s], dim=-1)
        alpha = torch.sigmoid(self.linear_LSTerm2(torch.sigmoid(self.linear_LSTerm1(concat_all))))
        final_user_embed = alpha * u_l + (1 - alpha) * u_s

        return final_user_embed, u_l, u_s, p_l, p_s

    def Interest_Predict(self, user_embed, targets):
        '''
        :param user_embed: shape(batch_size, dim)
        :param target_embed: shape(batch_size,) target item ID
        :return:
        '''
        user_embed = self.dropout(user_embed)
        test_item_emb = self.item_embeddings.weight[:self.args.item_size]
        logits = torch.matmul(user_embed, test_item_emb.transpose(0, 1))
        main_loss = self.criterion(logits, targets)
        return main_loss

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').            Unmasked positions are filled with float(0.0).        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -10000.0).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, data):
        """
        :param data: (3, #batch_size, ?) contain 3 tensor,
               data[0] = user_id, data[1] = action_seq(input) data[2] = target item
        :return:
        """

        user_ids = data[0]
        inputs = data[1]

        seq = inputs.flatten()
        seq_mask = (inputs == 0).float()
        seq_mask = 1.0 - seq_mask

        seq_hidden_global_a = self.gnn_encode(seq).view(-1, self.args.max_seq_length, self.args.hidden_size)
        seq_hidden_global_b = self.gnn_encode(seq).view(-1, self.args.max_seq_length, self.args.hidden_size)

        key_padding_mask = (inputs == 0)
        attn_mask = self.generate_square_subsequent_mask(self.args.max_seq_length).to(inputs.device)
        seq_embed_local = self.item_embeddings(inputs)
        seq_hidden_local = self.LayerNorm(seq_embed_local)
        seq_hidden_local = self.dropout(seq_hidden_local)
        seq_hidden_permute = seq_hidden_local.permute(1, 0, 2)

        encoded_layers = self.item_encoder(seq_hidden_permute,
                                           mask=attn_mask,
                                           src_key_padding_mask=key_padding_mask)

        sequence_output = encoded_layers.permute(1, 0, 2)

        short_user_embed = encoded_layers[-1].unsqueeze(0)

        short_embedding, _ = self.SeqLayer(seq_hidden_permute)
        u_s = self.ShortTermMHA(short_user_embed, short_embedding, short_embedding, key_padding_mask=seq_mask,
                                need_weights=False)[0].permute(1, 0, 2).squeeze()

        # user_embed = self.user_embeddings(user_ids).unsqueeze(0)
        # u_l_a, gating_score_a = self.LongTermMHA(user_embed, seq_hidden_global_a.permute(1, 0, 2), seq_hidden_global_a.permute(1, 0, 2),
        #                        key_padding_mask=key_padding_mask)
        # u_l_b, gating_score_b = self.LongTermMHA(user_embed, seq_hidden_global_b.permute(1, 0, 2), seq_hidden_global_b.permute(1, 0, 2),
        #                        key_padding_mask=key_padding_mask)
        # u_l_a, u_l_b, gating_score_a, gating_score_b = u_l_a.squeeze(), u_l_b.squeeze(), gating_score_a.squeeze(), gating_score_b.squeeze()
        # user_seq_a = seq_hidden_global_a * gating_score_a.unsqueeze(2)
        # user_seq_b = seq_hidden_global_b * gating_score_b.unsqueeze(2)

        user_emb = self.user_embeddings(user_ids).view(-1, self.args.hidden_size)
        gating_score_a = torch.sigmoid(torch.matmul(seq_hidden_global_a, self.gate_item.unsqueeze(0)).squeeze() +
                                       user_emb.mm(self.gate_user))
        user_seq_a = seq_hidden_global_a * gating_score_a.unsqueeze(2)
        gating_score_b = torch.sigmoid(torch.matmul(seq_hidden_global_b, self.gate_item.unsqueeze(0)).squeeze() +
                                       user_emb.mm(self.gate_user))
        user_seq_b = seq_hidden_global_b * gating_score_b.unsqueeze(2)

        user_seq_a = self.gnndrop(user_seq_a)
        user_seq_b = self.gnndrop(user_seq_b)

        # _, h_out = self.GRULayer(seq_hidden_permute, user_embed)
        # u_s = self.ShortTermMHA(h_out, encoded_layers, encoded_layers, key_padding_mask=key_padding_mask,
        #                         need_weights=False)[0].permute(1, 0, 2).squeeze()

        # design proxy label for LS-Term Interest representation
        # proxy_mask = seq_mask.unsqueeze(-1)
        # p_l = torch.sum(seq_embed_local * proxy_mask, dim=1) / torch.sum(proxy_mask, dim=1)
        #
        # position_mask = (torch.arange(0, proxy_mask.shape[1]) > (proxy_mask.shape[1] - self.args.ls_k - 1)).float().\
        #     unsqueeze(-1).to(proxy_mask.device)
        # short_proxy_mask = torch.logical_and(position_mask, proxy_mask).float()
        # p_s = torch.sum(seq_embed_local * short_proxy_mask, dim=1) / torch.sum(short_proxy_mask, dim=1)
        #
        all_seq_rep = torch.cat([user_seq_a, user_seq_b], dim=-1)
        all_seq_rep = self.linear_transform(all_seq_rep)

        # Adaptive Fusion for Interaction Prediction
        # _, final_state = self.GRULayer1(seq_hidden_local.permute(1, 0, 2))
        # concat_all = torch.cat([final_state.squeeze(), u_l_a, u_s], dim=-1)
        # alpha = torch.sigmoid(self.linear_LSTerm2(torch.sigmoid(self.linear_LSTerm1(concat_all))))
        # final_user_embed = alpha * u_l_a + (1 - alpha) * u_s

        # return final_user_embed, (seq_hidden_global_a, seq_hidden_global_b), (u_l_a, u_l_b, u_s, p_l, p_s), seq_mask
        return all_seq_rep, u_s, (user_seq_a, user_seq_b), (seq_hidden_global_a, seq_hidden_global_b), seq_mask

    def train_stage(self, data):
        user_ids, action_seq, targets = data[0], data[1], data[2]
        user_embed = self.user_embeddings(user_ids)
        seq_embed = self.item_embeddings(action_seq)
        target_embed = self.item_embeddings(targets)
        seq_mask = (action_seq == 0)
        final_user_embed, u_l, u_s, p_l, p_s = self.Interest_net(user_embed, seq_embed, target_embed, seq_mask, self.args.ls_k)
        main_loss = self.Interest_Predict(final_user_embed, targets)
        lsTerm_loss = self.LSTerm_loss(u_l, u_s, p_l, p_s, 1.-seq_mask.float(), self.args.ls_threshold)
        re_loss = torch.FloatTensor([0.0])
        joint_loss = main_loss + 0.1 * lsTerm_loss + 0.
        return joint_loss, main_loss, lsTerm_loss, re_loss

    def eval_stage(self, data):
        user_ids, action_seq, targets = data[0], data[1], data[2]
        user_embed = self.user_embeddings(user_ids)
        seq_embed = self.item_embeddings(action_seq)
        target_embed = self.item_embeddings(targets)
        seq_mask = (action_seq == 0)
        final_user_embed, _, _, _, _ = self.Interest_net(user_embed, seq_embed, target_embed, seq_mask)
        return final_user_embed


    def new_train_stage(self, data):
        uids, seq_ids, targets = data
        all_seq_rep, u_s, (user_seq_a, user_seq_b), (seq_gnn_a, seq_gnn_b), seq_mask = self.forward(data)
        u_l, fc = self.final_att_net(seq_mask.unsqueeze(-1), all_seq_rep)
        concat_all = torch.cat([u_l, u_s, fc], dim=-1)
        alpha = torch.tanh(self.linear_LSTerm2(torch.sigmoid(self.linear_LSTerm1(concat_all))))
        final_user_embed = alpha * u_l + (1 - alpha) * u_s

        seq_out = self.dropout(final_user_embed)
        test_item_emb = self.item_embeddings.weight[:self.args.item_size]
        logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        main_loss = self.criterion(logits, targets)

        sum_a = torch.sum(seq_gnn_a * seq_mask.unsqueeze(-1), 1) / torch.sum(seq_mask.unsqueeze(-1).float(), 1)
        sum_b = torch.sum(seq_gnn_b * seq_mask.unsqueeze(-1), 1) / torch.sum(seq_mask.unsqueeze(-1).float(), 1)

        info_hidden = torch.cat([sum_a, sum_b], 0)
        gcl_loss = self.GCL_loss(info_hidden, hidden_norm=True, temperature=0.5)


        p_l, p_s = self.generate_proxy(seq_ids)

        # [B, L, d] to [B, L]️, can reduce training time and memory
        if self.args.fast_run:
            seq_hidden_local = self.w_e(self.item_embeddings(data[1])).squeeze().unsqueeze(0)
            user_seq_a = self.w_g(user_seq_a).squeeze()
            user_seq_b = self.w_g(user_seq_b).squeeze()
        else:
            seq_hidden_local = self.item_embeddings(data[1])
        mmd_loss = self.MMD_loss(seq_hidden_local, user_seq_a) + self.MMD_loss(seq_hidden_local, user_seq_b)



        lsTerm_loss = self.LSTerm_loss(u_l, u_s, p_l, p_s, seq_mask, self.args.ls_threshold)
        # lsTerm_loss_b = self.LSTerm_loss(u_l_b, u_s, p_l, p_s, seq_mask, self.args.ls_threshold)
        # lsTerm_loss = lsTerm_loss_a + lsTerm_loss_b
        # re_loss = torch.FloatTensor([0.0])

        joint_loss = main_loss + self.args.lam1 * gcl_loss + self.args.lam2 * (mmd_loss + lsTerm_loss)

        return joint_loss, main_loss, gcl_loss, lsTerm_loss, mmd_loss

    def new_eval_stage(self, data):
        all_seq_rep, u_s, _, _, seq_mask = self.forward(data)
        u_l, fc = self.final_att_net(seq_mask.unsqueeze(-1), all_seq_rep)
        concat_all = torch.cat([u_l, u_s, fc], dim=-1)
        alpha = torch.tanh(self.linear_LSTerm2(torch.sigmoid(self.linear_LSTerm1(concat_all))))
        final_user_embed = alpha * u_l + (1 - alpha) * u_s
        return final_user_embed


    def New_Interest_net(self, user_ids, augment_seq_embed, seq_encoder_embed, orignal_seq_emb, seq_mask):
        '''
        :param user_embedding: shape(batch_size, dim)
        :param action_embedding: shape(batch_size, seq_len, dim)
        :param target_embedding: shape(batch_size, dim)
        :param seq_mask: shape(batch_size, seq_len)  =1 padding  =0 real data
        :param k: short interest's sequence length
        :return: shape(batch_size, dim)
        '''

        # calculate user's LS-Term Interest
        long_user_embed = self.long_user_embeddings(user_ids).unsqueeze(0)

        u_l = self.LongTermMHA(long_user_embed, augment_seq_embed.permute(1, 0, 2), augment_seq_embed.permute(1, 0, 2),
                               key_padding_mask=seq_mask, need_weights=False)[0].permute(1, 0, 2).squeeze()

        short_user_embed = seq_encoder_embed[-1].unsqueeze(0)

        short_embedding, _ = self.SeqLayer(orignal_seq_emb.permute(1, 0, 2))
        u_s = self.ShortTermMHA(short_user_embed, short_embedding, short_embedding, key_padding_mask=seq_mask,
                                need_weights=False)[0].permute(1, 0, 2).squeeze()

        long_user_embed = long_user_embed.squeeze()
        short_user_embed = short_user_embed.squeeze()


        # # Adaptive Fusion for Interaction Prediction
        _, final_state = self.GRULayer1(orignal_seq_emb.permute(1, 0, 2))
        # concat_all = torch.cat([final_state.squeeze(), target_embedding, u_l, u_s], dim=-1)
        concat_all = torch.cat([final_state.squeeze(), u_l, u_s], dim=-1)
        alpha = torch.sigmoid(self.linear_LSTerm2(torch.sigmoid(self.linear_LSTerm1(concat_all))))
        final_user_embed = alpha * u_l + (1 - alpha) * u_s

        return final_user_embed, u_l, u_s, long_user_embed, short_user_embed

    def generate_proxy(self, seq_ids):
        max_seq_len = seq_ids.shape[-1]
        action_embed = self.item_embeddings(seq_ids)
        seq_mask = (1 - (seq_ids == 0).float()).unsqueeze(-1)
        pos_mask = (torch.arange(0, max_seq_len) > (max_seq_len - 1 - self.args.ls_k)).float().unsqueeze(-1).to(self.device)
        short_mask = torch.logical_and(pos_mask, seq_mask).float()

        p_l, fc = self.final_att_net(seq_mask, action_embed)

        # p_l = torch.sum(seq_mask * action_embed, dim=1) / torch.sum(seq_mask, dim=1)
        p_s = torch.sum(short_mask * action_embed, dim=1) / torch.sum(short_mask, dim=1)

        return p_l, p_s










    # def LSTerm_loss(self, u_l, u_s, p_l, p_s, seq_mask, threshold=3):
    #     '''
    #     :param u_l: user long term interest | shape(batch_size, dim)
    #     :param u_s: user short term interst | shape(batch_size, dim)
    #     :param p_l: user long interest proxy | shape(batch_size, dim)
    #     :param p_s: user short interst proxy | shape(batch_size, dim)
    #     :param seq_mask: user sequence padding mask | shape(batch_size, seq_len)
    #     :param threshold: consider ls-term interst's seq length
    #     :return: mean_loss
    #     '''
    #
    #     seq_len = torch.sum(seq_mask, dim=-1)
    #     LSTerm_mask = torch.where(torch.greater(seq_len, threshold), torch.ones_like(seq_len), torch.zeros_like(seq_len))
    #     long_mean_recent_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(u_l * (-p_l + p_s), dim=-1))) / torch.sum(seq_len, dim=-1)
    #     short_recent_mean_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(u_s * (-p_s + p_l), dim=-1))) / torch.sum(seq_len, dim=-1)
    #     mean_long_short_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(p_l * (-u_l + u_s), dim=-1))) / torch.sum(seq_len, dim=-1)
    #     recent_short_long_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(p_s * (-u_s+ u_l), dim=-1))) / torch.sum(seq_len, dim=-1)
    #     lsTerm_loss = long_mean_recent_loss + short_recent_mean_loss + mean_long_short_loss + recent_short_long_loss
    #     return lsTerm_loss
    #
    # def Disc_loss(self, short_user_embed, long_user_embed):
    #     discrepancy_loss = self.mse(short_user_embed, long_user_embed)
    #     # discrepancy_loss = (1/2) * (long_user_embed.norm(2).pow(2)) / float(long_user_embed.shape[0])
    #     return discrepancy_loss
    #
    # def forward(self, data):
    #     # calculate user's LS-Term Interest
    #     user_ids, action_seq = data[0], data[1]
    #     long_user_embed = self.long_user_embeddings(user_ids).unsqueeze(0)
    #     short_user_embed = self.long_user_embeddings(user_ids).unsqueeze(0)
    #
    #     action_embedding = self.item_embeddings(action_seq)
    #     # action_embed = self.LayerNorm(action_embedding)
    #     # action_embed = self.dropout(action_embed).permute(1, 0, 2)
    #
    #     seq_mask = (action_seq == 0)
    #
    #     u_l = self.LongTermMHA(long_user_embed, action_embedding.permute(1, 0, 2), action_embedding.permute(1, 0, 2),
    #                            key_padding_mask=seq_mask, need_weights=False)[0].permute(1, 0, 2).squeeze()
    #     _, h_out = self.GRULayer(action_embedding.permute(1, 0, 2), short_user_embed)
    #     # h0 = torch.zeros(1, action_embedding.shape[0], self.args.hidden_size).to(self.device)
    #     short_embedding, _ = self.SeqLayer(action_embedding.permute(1, 0, 2))
    #     u_s = self.ShortTermMHA(h_out, short_embedding, short_embedding, key_padding_mask=seq_mask,
    #                             need_weights=False)[0].permute(1, 0, 2).squeeze()
    #     long_user_embed = long_user_embed.squeeze()
    #     short_user_embed = short_user_embed.squeeze()
    #
    #
    #     # design proxy label for LS-Term Interest representation
    #     proxy_mask = (1. - seq_mask.float()).unsqueeze(-1)
    #     p_l = torch.sum(action_embedding * proxy_mask, dim=1) / torch.sum(proxy_mask, dim=1)
    #
    #     position_mask = (torch.arange(0, proxy_mask.shape[1]) > (proxy_mask.shape[1] - self.args.ls_k - 1)).float().unsqueeze(-1)\
    #         .to(proxy_mask.device)
    #     short_proxy_mask = torch.logical_and(position_mask, proxy_mask).float()
    #     p_s = torch.sum(action_embedding * short_proxy_mask, dim=1) / torch.sum(short_proxy_mask, dim=1)
    #
    #     # # Adaptive Fusion for Interaction Prediction
    #     _, final_state = self.GRULayer1(action_embedding.permute(1, 0, 2))
    #     # concat_all = torch.cat([final_state.squeeze(), target_embedding, u_l, u_s], dim=-1)
    #     concat_all = torch.cat([final_state.squeeze(), u_l, u_s], dim=-1)
    #
    #     # alpha = torch.sigmoid(self.linear_LSTerm2(torch.sigmoid(self.linear_LSTerm1(concat_all))))
    #     # final_user_embed = alpha * u_l + (1 - alpha) * u_s
    #     # final_user_embed = torch.tanh(self.linear_LSTerm1(concat_all))
    #     alpha = torch.sigmoid(self.linear_LSTerm2(torch.sigmoid(self.linear_LSTerm1(concat_all))))
    #     final_user_embed = alpha * u_l + (1 - alpha) * u_s
    #
    #     return final_user_embed, u_l, u_s, p_l, p_s, seq_mask, short_user_embed, long_user_embed
    #
    # def train_stage(self, data):
    #     targets = data[2]
    #     final_user_embed, u_l, u_s, p_l, p_s, seq_mask, short_user_embed, long_user_embed = self.forward(data)
    #     final_user_embed = self.dropout(final_user_embed)
    #     test_item_emb = self.item_embeddings.weight[:self.args.item_size]
    #     logits = torch.matmul(final_user_embed, test_item_emb.transpose(0, 1))
    #     main_loss = self.criterion(logits, targets)
    #     lsTerm_loss = self.LSTerm_loss(u_l, u_s, p_l, p_s, 1.-seq_mask.float())
    #     disc_loss = self.Disc_loss(short_user_embed, long_user_embed)
    #     joint_loss = main_loss + lsTerm_loss + 0.
    #     return joint_loss, main_loss, lsTerm_loss, disc_loss
    #
    # def eval_stage(self, data):
    #     final_user_embed, u_l, u_s, _, _, _, _, _ = self.forward(data)
    #     return final_user_embed
    #
    #

class CL4SR(nn.Module):
    def __init__(self, args, global_graph):
        super(CL4SR, self).__init__()
        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda:{}".format(self.args.gpu_id) if self.cuda_condition else "cpu")
        self.global_graph = global_graph.to(self.device)
        self.global_gnn = GlobalGNN(args)

        self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size)
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        # sequence encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size,
                                                        nhead=args.num_attention_heads,
                                                        dim_feedforward=4 * args.hidden_size,
                                                        dropout=args.attention_probs_dropout_prob,
                                                        activation=args.hidden_act)
        self.item_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.num_hidden_layers)

        # InterestAtten
        self.short_user_embeddings = nn.Embedding(args.user_size, args.hidden_size)
        self.LongTermMHA = nn.MultiheadAttention(embed_dim=args.hidden_size, num_heads=1, dropout=0.1)
        self.ShortTermMHA = nn.MultiheadAttention(embed_dim=args.hidden_size, num_heads=1, dropout=0.1)
        self.GRULayer = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=1)
        self.SeqLayer = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=1)
        self.GRULayer1 = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=1)
        self.linear_LSTerm1 = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.linear_LSTerm2 = nn.Linear(args.hidden_size, 1)


        # AttNet
        self.w_1 = nn.Parameter(torch.Tensor(2 * args.hidden_size, args.hidden_size))
        self.w_2 = nn.Parameter(torch.Tensor(args.hidden_size, 1))
        self.linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_2 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        # fast run with mmd
        self.w_g = nn.Linear(args.hidden_size, 1)
        self.w_e = nn.Linear(args.hidden_size, 1)

        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.linear_transform = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        self.gnndrop = nn.Dropout(args.gnn_dropout_prob)

        self.criterion = nn.CrossEntropyLoss()
        self.betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=self.betas,
                                          weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.args = args
        self.apply(self._init_weights)

        # user-specific gating
        self.gate_item = Variable(torch.zeros(args.hidden_size, 1).type
                                  (torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_user = Variable(torch.zeros(args.hidden_size, args.max_seq_length).type
                                  (torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_item = torch.nn.init.xavier_uniform_(self.gate_item)
        self.gate_user = torch.nn.init.xavier_uniform_(self.gate_user)

    def _init_weights(self, module):
        """ Initialize the weights """
        stdv = 1.0 / math.sqrt(self.args.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def MMD_loss(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        if self.args.fast_run:
            source = source.view(-1, self.args.max_seq_length)
            target = target.view(-1, self.args.max_seq_length)
            batch_size = int(source.size()[0])
            loss_all = []
            kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                           fix_sigma=fix_sigma)
            xx = kernels[:batch_size, :batch_size]
            yy = kernels[batch_size:, batch_size:]
            xy = kernels[:batch_size, batch_size:]
            yx = kernels[batch_size:, :batch_size]
            loss = torch.mean(xx + yy - xy - yx)
            loss_all.append(loss)
        else:
            source = source.view(-1, self.args.max_seq_length, self.args.hidden_size)
            target = target.view(-1, self.args.max_seq_length, self.args.hidden_size)
            batch_size = int(source.size()[1])
            loss_all = []
            for i in range(int(source.size()[0])):
                kernels = self.guassian_kernel(source[i], target[i], kernel_mul=kernel_mul, kernel_num=kernel_num,
                                               fix_sigma=fix_sigma)
                xx = kernels[:batch_size, :batch_size]
                yy = kernels[batch_size:, batch_size:]
                xy = kernels[:batch_size, batch_size:]
                yx = kernels[batch_size:, :batch_size]
                loss = torch.mean(xx + yy - xy - yx)
                loss_all.append(loss)
        return sum(loss_all) / len(loss_all)

    def GCL_loss(self, hidden, hidden_norm=True, temperature=1.0):
        batch_size = hidden.shape[0] // 2
        LARGE_NUM = 1e9
        # inner dot or cosine
        if hidden_norm:
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        hidden_list = torch.split(hidden, batch_size, dim=0)
        hidden1, hidden2 = hidden_list[0], hidden_list[1]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.from_numpy(np.arange(batch_size)).to(hidden.device)
        masks = torch.nn.functional.one_hot(torch.from_numpy(np.arange(batch_size)).to(hidden.device), batch_size)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(1, 0)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(1, 0)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(1, 0)) / temperature
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(1, 0)) / temperature

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
        loss = (loss_a + loss_b)
        return loss

    def gnn_encode(self, items):
        subgraph_loaders = NeighborSampler(self.global_graph.edge_index, node_idx=items, sizes=self.args.sample_size,
                                           shuffle=False,
                                           num_workers=0, batch_size=items.shape[0])
        g_adjs = []
        s_nodes = []
        for (b_size, node_idx, adjs) in subgraph_loaders:
            if type(adjs) == list:
                g_adjs = [adj.to(items.device) for adj in adjs]
            else:
                g_adjs = adjs.to(items.device)
            n_idxs = node_idx.to(items.device)
            s_nodes = self.item_embeddings(n_idxs).squeeze()
        attr = self.global_graph.edge_attr.to(items.device)
        g_hidden = self.global_gnn(s_nodes, g_adjs, attr)
        return g_hidden

    def final_att_net(self, seq_mask, hidden):
        batch_size = hidden.shape[0]
        lens = hidden.shape[1]
        pos_emb = self.position_embeddings.weight[:lens]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        seq_hidden = torch.sum(hidden * seq_mask, -2) / torch.sum(seq_mask, 1)
        seq_hidden = seq_hidden.unsqueeze(-2).repeat(1, lens, 1)
        item_hidden = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        item_hidden = torch.tanh(item_hidden)
        score = torch.sigmoid(self.linear_1(item_hidden) + self.linear_2(seq_hidden))
        att_score = torch.matmul(score, self.w_2)
        att_score_masked = att_score * seq_mask
        output = torch.sum(att_score_masked * hidden, 1)
        return output

    def LSTerm_loss(self, u_l, u_s, p_l, p_s, seq_mask, threshold=3, margin=1.0):
        '''
        :param u_l: user long term interest | shape(batch_size, dim)
        :param u_s: user short term interst | shape(batch_size, dim)
        :param p_l: user long interest proxy | shape(batch_size, dim)
        :param p_s: user short interst proxy | shape(batch_size, dim)
        :param seq_mask: user sequence padding mask | shape(batch_size, seq_len)
        :param threshold: consider ls-term interst's seq length
        :return: mean_loss
        '''

        seq_len = torch.sum(seq_mask, dim=-1)
        LSTerm_mask = torch.where(torch.greater(seq_len, threshold), torch.ones_like(seq_len), torch.zeros_like(seq_len))
        if self.args.lsTerm_loss == 'bpr':
            long_mean_recent_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(u_l * (-p_l + p_s), dim=-1))) / torch.sum(LSTerm_mask)
            short_recent_mean_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(u_s * (-p_s + p_l), dim=-1))) / torch.sum(LSTerm_mask)
            mean_long_short_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(p_l * (-u_l + u_s), dim=-1))) / torch.sum(LSTerm_mask)
            recent_short_long_loss = torch.sum(LSTerm_mask * F.softplus(torch.sum(p_s * (-u_s+ u_l), dim=-1))) / torch.sum(LSTerm_mask)
        elif self.args.lsTerm_loss == 'triplet':
            distance_long_mean = torch.square_(u_l - p_l)
            distance_long_recent = torch.square_(u_l - p_s)
            distance_short_mean = torch.square_(u_s - p_l)
            distance_short_recent = torch.square_(u_s - p_s)
            long_mean_recent_loss = torch.sum(
                LSTerm_mask * torch.sum(torch.maximum(0.0, distance_long_mean - distance_long_recent + margin),
                                        dim=-1)) / torch.sum(LSTerm_mask)
            short_recent_mean_loss = torch.sum(
                LSTerm_mask * torch.sum(torch.maximum(0.0, distance_short_recent - distance_short_mean + margin),
                                        dim=-1)) / torch.sum(LSTerm_mask)
            mean_long_short_loss = torch.sum(
                LSTerm_mask * torch.sum(torch.maximum(0.0, distance_long_mean - distance_short_mean + margin),
                                        dim=-1)) / torch.sum(LSTerm_mask)
            recent_short_long_loss = torch.sum(
                LSTerm_mask * torch.sum(torch.maximum(0.0, distance_short_recent - distance_long_recent + margin),
                                        dim=-1)) / torch.sum(LSTerm_mask)
        lsTerm_loss = long_mean_recent_loss + short_recent_mean_loss + mean_long_short_loss + recent_short_long_loss
        return lsTerm_loss

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').            Unmasked positions are filled with float(0.0).        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -10000.0).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, data):
        """
        :param data: (3, #batch_size, ?) contain 3 tensor,
               data[0] = user_id, data[1] = action_seq(input) data[2] = target item
        :return:
        """

        user_ids = data[0]
        inputs = data[1]

        seq = inputs.flatten()
        seq_mask = (inputs == 0).float().unsqueeze(-1)
        seq_mask = 1.0 - seq_mask

        seq_hidden_global_a = self.gnn_encode(seq).view(-1, self.args.max_seq_length, self.args.hidden_size)
        seq_hidden_global_b = self.gnn_encode(seq).view(-1, self.args.max_seq_length, self.args.hidden_size)

        key_padding_mask = (inputs == 0)
        # attn_mask = self.generate_square_subsequent_mask(self.args.max_seq_length).to(inputs.device)
        seq_hidden_local = self.item_embeddings(inputs)
        seq_hidden_local = self.LayerNorm(seq_hidden_local)
        seq_hidden_local = self.dropout(seq_hidden_local)
        seq_hidden_permute = seq_hidden_local.permute(1, 0, 2)
        #
        # encoded_layers = self.item_encoder(seq_hidden_permute,
        #                                    mask=attn_mask,
        #                                    src_key_padding_mask=key_padding_mask)
        #
        # sequence_output = encoded_layers.permute(1, 0, 2)

        short_user_embed = self.short_user_embeddings(user_ids).unsqueeze(0)
        _, h_out = self.GRULayer(seq_hidden_permute, short_user_embed)
        h0 = torch.zeros(1, seq_hidden_local.shape[0], self.args.hidden_size).to(self.device)
        short_embedding, _ = self.SeqLayer(seq_hidden_permute, h0)
        u_s = self.ShortTermMHA(h_out, short_embedding, short_embedding, key_padding_mask=key_padding_mask,
                                need_weights=False)[0].permute(1, 0, 2).squeeze()

        user_emb = self.user_embeddings(user_ids).view(-1, self.args.hidden_size)

        gating_score_a = torch.sigmoid(torch.matmul(seq_hidden_global_a, self.gate_item.unsqueeze(0)).squeeze() +
                                       user_emb.mm(self.gate_user))
        user_seq_a = seq_hidden_global_a * gating_score_a.unsqueeze(2)
        gating_score_b = torch.sigmoid(torch.matmul(seq_hidden_global_b, self.gate_item.unsqueeze(0)).squeeze() +
                                       user_emb.mm(self.gate_user))
        user_seq_b = seq_hidden_global_b * gating_score_b.unsqueeze(2)

        user_seq_a = self.gnndrop(user_seq_a)
        user_seq_b = self.gnndrop(user_seq_b)

        # hidden = torch.cat([sequence_output, user_seq_a, user_seq_b], -1)
        hidden = torch.cat([user_seq_a, user_seq_a], -1)
        hidden = self.linear_transform(hidden)

        return hidden, (seq_hidden_global_a, seq_hidden_global_b), u_s, seq_mask

    def new_train_stage(self, data):
        seq_ids, targets = data[1], data[2]
        hidden, (seq_gnn_a, seq_gnn_b), u_s, seq_mask = self.forward(data)
        p_l, p_s = self.generate_proxy(seq_ids)

        u_l = self.final_att_net(seq_mask, hidden)

        concat_all = torch.cat([u_l, u_s], dim=-1)
        alpha = torch.tanh(self.linear_LSTerm2(torch.sigmoid(self.linear_LSTerm1(concat_all))))
        final_user_embed = alpha * u_l + (1 - alpha) * u_s
        final_user_embed = self.dropout(final_user_embed)
        # print(seq_out.shape)
        test_item_emb = self.item_embeddings.weight[:self.args.item_size]
        logits = torch.matmul(final_user_embed, test_item_emb.transpose(0, 1))
        main_loss = self.criterion(logits, targets)

        sum_a = torch.sum(seq_gnn_a * seq_mask, 1) / torch.sum(seq_mask.float(), 1)
        sum_b = torch.sum(seq_gnn_b * seq_mask, 1) / torch.sum(seq_mask.float(), 1)

        info_hidden = torch.cat([sum_a, sum_b], 0)
        gcl_loss = self.GCL_loss(info_hidden, hidden_norm=True, temperature=0.5)
        #
        # # [B, L, d] to [B, L]️, can reduce training time and memory
        # if self.args.fast_run:
        #     seq_hidden_local = self.w_e(self.item_embeddings(data[1])).squeeze().unsqueeze(0)
        #     user_seq_a = self.w_g(user_seq_a).squeeze()
        #     user_seq_b = self.w_g(user_seq_b).squeeze()
        # else:
        #     seq_hidden_local = self.item_embeddings(data[1])
        # mmd_loss = self.MMD_loss(seq_hidden_local, user_seq_a) + self.MMD_loss(seq_hidden_local, user_seq_b)

        # gcl_loss = torch.tensor([0.0]).to(self.device)
        lst_loss = self.LSTerm_loss(u_l, u_s, p_l, p_s, seq_mask.squeeze())
        mmd_loss = torch.tensor([0.0]).to(self.device)

        joint_loss = main_loss + self.args.lam1 * gcl_loss + self.args.lam2 * (mmd_loss + lst_loss)

        return joint_loss, main_loss, gcl_loss, lst_loss, mmd_loss



    def new_eval_stage(self, data):
        hidden, _, u_s, seq_mask = self.forward(data)
        u_l = self.final_att_net(seq_mask, hidden)

        concat_all = torch.cat([u_l, u_s], dim=-1)
        alpha = torch.tanh(self.linear_LSTerm2(torch.sigmoid(self.linear_LSTerm1(concat_all))))
        final_user_embed = alpha * u_l + (1 - alpha) * u_s

        return final_user_embed

    def generate_proxy(self, seq_ids):
        max_seq_len = seq_ids.shape[-1]
        action_embed = self.item_embeddings(seq_ids)
        seq_mask = (1 - (seq_ids == 0).float()).unsqueeze(-1)
        pos_mask = (torch.arange(0, max_seq_len) > (max_seq_len - 1 - self.args.ls_k)).float().unsqueeze(-1).to(
            self.device)
        short_mask = torch.logical_and(pos_mask, seq_mask).float()

        p_l = torch.sum(seq_mask * action_embed, dim=1) / torch.sum(seq_mask, dim=1)
        p_s = torch.sum(short_mask * action_embed, dim=1) / torch.sum(short_mask, dim=1)

        return p_l, p_s