#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.nn.init as init
from numba import jit
from entmax import entmax_bisect
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from recbole.config import Config
from mamba_ssm import Mamba  # 从mamba_ssm块中导入一个Mamba函数
from temporal_attention_layer import TA_layer

# 用于实现层归一化
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)  # 计算输入x的均值
        # 计算x的方差
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # 归一化：使x均值为0，标准差为1
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


# GCN定义
# 一般输入为（batch_siza，input_size)
class GraphConvolution(Module):
    # (输入特征数，输出特征数，是否有偏置）
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features * 2, out_features * 2))  # 维度从in变为out
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features * 2))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # （给权重和偏置赋予初始值）
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 前向传播（输入数据，邻接矩阵）# [b,s,d]#[batch_size, batch_size]
    def forward(self, input, adj):
        # print(input.size())
        # print(self.weight.size())
        # print(input.size())
        batch_size, seq_len, de_size = input.size()
        # 将输入张量形状重塑
        support = input.view(batch_size * seq_len, -1).matmul(self.weight)
        support = support.view(batch_size, seq_len, -1)  # 恢复batch维度
        # print(support.size())
        # print(adj.size())
        # 应用邻接矩阵进行稀疏矩阵乘法
        # output = torch.matmul(support, adj)
        output = torch.matmul(adj, support)

        # support = torch.mm(input, self.weight)  # 输出维度由self.weight的列数决定
        # output = torch.spmm(adj, support)
        if self.bias is not None:  # bias不影响维度变化
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GNN(nn.Module):
    # self.gnn = GNN(self.hidden_size, 0.5)
    # def __init__(self, batch, hidden, dropout):
    def __init__(self, hidden, dropout):
        super(GNN, self).__init__()
        # 定义第一层图卷积
        self.gc1 = GraphConvolution(hidden, hidden)
        # 定义第二层图卷积
        self.gc2 = GraphConvolution(hidden, hidden)
        # 时间注意力机制
        self.TA = TA_layer(hidden*2, hidden*2, 2, 2)
        self.dropout = dropout
        # self.batch = batch

    # hidden = self.gnn(seq_emb, A)  # (b,s,2d)
    def forward(self, x, adj):
        # for batch_index in range(x.size(0)):  # 遍历 batch 维度
        # 获取当前 batch 的 hidden x hidden 矩阵
        # batch_matrix = x[batch_index]
        # 第一层卷积后＋ReLU激活函数
        x = F.relu(self.gc1(x, adj))
        # 利用dropout
        x = F.dropout(x, self.dropout, training=self.training)
        # 第二层卷积
        x = self.gc2(x, adj)
         # 时间注意力机制
        x = self.TA(x)
        return x


# class Attention(nn.Module):
#     # （输入维度，隐层维度）
#     def __init__(self, in_size, hidden_size=200):
#         super(Attention, self).__init__()
#         # 序列模型（两个线性层＋Tanh激活函数）
#         self.project = nn.Sequential(
#             nn.Linear(in_size * 2, hidden_size),
#             nn.Tanh(),
#             nn.Linear(hidden_size, 1, bias=False)
#         )
#
#     def forward(self, z):
#         # print(z.size())
#         w = self.project(z)
#         # print(w.size())
#         # 获得注意力权重
#         beta = torch.softmax(w, dim=1)
#         return (beta * z).sum(1), beta
#
#
# class GNN(nn.Module):
#     def __init__(self, hidden_size, dropout):
#         super(GNN, self).__init__()
#         # 三层卷积网络的定义
#         # self.gnn = GNN(self.hidden_size, 0.5)
#         self.dropout = dropout
#         self.SGCN1 = GCN(hidden_size, dropout)
#         self.SGCN2 = GCN(hidden_size, dropout)
#         # （内容卷积网络）
#         self.CGCN = GCN(hidden_size, dropout)
#
#         # 定义注意力参数
#         self.a = nn.Parameter(torch.zeros(size=(hidden_size, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#         # 注意力层
#         self.attention = Attention(hidden_size)
#         self.tanh = nn.Tanh()
#         # 序列模型的定义
#         self.MLP = nn.Sequential(
#             nn.Linear(hidden_size * 2, hidden_size * 2),
#             nn.LogSoftmax(dim=1)
#         )
#
#     def forward(self, x, adj):
#         # print(x.size())
#         emb1 = self.SGCN1(x, adj)
#         com1 = self.CGCN(x, adj)
#         com2 = self.CGCN(x, adj)
#         emb2 = self.SGCN2(x, adj)
#         Xcom = (com1 + com2) / 2
#         # print(emb2.size())
#         # print(com1.size())
#         emb = torch.stack([emb1, emb2, Xcom], dim=1)
#         # print('666')
#         # print(emb.size())
#         # 通过注意力层
#         emb, att = self.attention(emb)
#         # print(emb.size())
#         # print(emb.size())
#         output = self.MLP(emb)
#         # print(output.size())
#         return output
#         # return output, att, emb1, com1, com2, emb2, emb
#

# # 查找邻居节点
class FindNeighbors(Module):
    def __init__(self, hidden_size,nei_n):
        super(FindNeighbors, self).__init__()
        self.hidden_size = hidden_size
        self.neighbor_n = nei_n # Diginetica:3; Tmall: 7; Nowplaying: 4
        self.dropout40 = nn.Dropout(0.40)

    # 计算会话嵌入向量之间的相似度
    def compute_sim(self, sess_emb):
        fenzi = torch.matmul(sess_emb, sess_emb.permute(1, 0))
        fenmu_l = torch.sum(sess_emb * sess_emb + 0.000001, 1)
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)
        fenmu = torch.matmul(fenmu_l, fenmu_l.permute(1, 0))
        cos_sim = fenzi / fenmu
        cos_sim = nn.Softmax(dim=-1)(cos_sim)
        return cos_sim

    def forward(self, sess_emb):
        k_v = self.neighbor_n  # 存储邻居的数量
        cos_sim = self.compute_sim(sess_emb)  # 计算相似度
        # 如果会话数量小于邻居数，则调整邻居数
        if cos_sim.size()[0] < k_v:
            k_v = cos_sim.size()[0]
        # 使用topk方法找到每个会话的k个最相似邻居的相似度
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1)
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        # 根据索引获取每个会话k个最相似邻居的嵌入向量
        sess_topk = sess_emb[topk_indice]

        cos_sim = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.hidden_size)
        #  计算邻居会话的加权和
        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)
        # 应用dropout层
        neighbor_sess = self.dropout40(neighbor_sess)  # [b,d]
        # 返回经过dropout处理的会话嵌入向量
        return neighbor_sess


# mamba模块
class Mamba4Rec(nn.Module):
    # def __init__(self, opt, config, dataset):
    def __init__(self, opt, config):
        super(Mamba4Rec, self).__init__()
        # 从配置中获取模型的超参数
        self.hidden_size = opt.hiddenSize
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        # self.hidden_size = config["hidden_size"]
        # self.loss_type = config["loss_type"]
        # self.num_layers = config["num_layers"]
        # self.dropout_prob = config["dropout_prob"]

        # mamba块的超参数
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        # 定义嵌入层
        # 将物品ID转换为嵌入向量
        # self.item_embedding = nn.Embedding(
        #     self.n_items, self.hidden_size, padding_idx=0
        # )
        # 层归一化
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        # dropout
        self.dropout = nn.Dropout(self.dropout_prob)

        # 定义mamba层列表：包含多个mambalayer
        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                # d_model=self.hidden_size,
                d_model=200,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])

        # 根据损失类型定义损失函数
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    # 权重初始化
    def _init_weights(self, module):
        # 如果模块是线性层或嵌入层，进行权重初始化
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        # 如果是层归一化，偏置为0，权重为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果是带偏置的线性层，初始化偏置为0
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1]).to(output.device)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, item_emb, item_seq_len):
        # def forward(self, item_seq, item_seq_len):
        # 节点嵌入
        # item_emb = self.item_embedding(item_seq)
        # item_emb = self.dropout(item_emb)
        # item_emb = self.LayerNorm(item_emb)

        # 通过mamba层前向传播
        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)
        c = item_emb[:, -1, :].unsqueeze(1)  # [b,d]->[b,1,d]
        x_n = item_emb[:, :-1, :]  # [b,s,d]
        # # 根据序列长度获取输出
        # seq_output = self.gather_indexes(item_emb, item_seq_len - 1)

        return c, x_n


class MambaLayer(nn.Module):
    # 初始化块的参数及层
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # 创建mamba模块
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # 创建dropout层
        self.dropout = nn.Dropout(dropout)
        # 创建层归一化
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        # 创建前馈神经网络
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    # 前向传播方法
    def forward(self, input_tensor):
        # 调用mamba块处理张量得到隐藏状态
        hidden_states = self.mamba(input_tensor)

        # 判断mamba层数
        if self.num_layers == 1:  # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:  # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        # 通过前馈神经网络处理隐藏状态
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        # 创建两个线性层
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        # 创建激活函数
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        # 依次通过第一个线性层，激活函数，dropout
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # 依次通过第二个线性层，dropout，层归一化和残差连接
        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class RelationGAT(Module):
    def __init__(self, batch_size, hidden_size=100):
        super(RelationGAT, self).__init__()
        self.batch_size = batch_size
        self.dim = hidden_size
        # 定义一个线性层，输入维度为2倍的隐藏层维度，输出维度为隐藏层维度
        self.w_f = nn.Linear(2 * hidden_size, hidden_size)
        # 计算注意力权重参数
        self.alpha_w = nn.Linear(self.dim, 1)
        # 注意力权重参数矩阵
        self.atten_w0 = nn.Parameter(torch.Tensor(1, self.dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_bias = nn.Parameter(torch.Tensor(self.dim))

    # 获取注意力权重
    def get_alpha(self, x=None):
        # x[b,1,d]
        alpha_global = torch.sigmoid(self.alpha_w(x)) + 1  # [b,1,1]
        alpha_global = self.add_value(alpha_global)
        return alpha_global  # [b,1,1]

    # 对输入value进行处理
    def add_value(self, value):
        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        return value

    # 计算目标张量和键值对张量之间的注意力
    def tglobal_attention(self, target, k, v, alpha_ent=1):
        # 计算权重
        alpha = torch.matmul(torch.relu(k.matmul(self.atten_w1) + target.matmul(self.atten_w2) + self.atten_bias),
                             self.atten_w0.t())
        # 调用entmax函数调整alpha值
        alpha = entmax_bisect(alpha, alpha_ent, dim=1)
        c = torch.matmul(alpha.transpose(1, 2), v)
        return c

    def forward(self, item_embedding, items, A, D, target_embedding):
        seq_h = []
        for i in torch.arange(items.shape[0]):
            seq_h.append(torch.index_select(item_embedding, 0, items[i]))  # [b,s,d]
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        len = seq_h1.shape[1]  # 获取序列长度
        relation_emb_gcn = torch.sum(seq_h1, 1)  # [b,d]
        DA = torch.mm(D, A).float()  # [b,b]
        relation_emb_gcn = torch.mm(DA, relation_emb_gcn)  # [b,d]
        relation_emb_gcn = relation_emb_gcn.unsqueeze(1).expand(relation_emb_gcn.shape[0], len,
                                                                relation_emb_gcn.shape[1])  # [b,s,d]

        target_emb = self.w_f(target_embedding)

        alpha_line = self.get_alpha(x=target_emb)
        q = target_emb  # [b,1,d]
        k = relation_emb_gcn  # [b,1,d]
        v = relation_emb_gcn  # [b,1,d]

        # 调用tglobal_attention方法计算注意力加权:SparseTargetAttentionMechanism
        line_c = self.tglobal_attention(q, k, v, alpha_ent=alpha_line)  # [b,1,d]
        # 使用selu激活函数并去除单维度
        c = torch.selu(line_c).squeeze()
        # 标准化
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))

        return l_c  # [b,d]


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.dataset = opt.dataset
        self.hidden_size = opt.hiddenSize
        self.nei_n = opt.nei_n
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        # 节点初始化嵌入
        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0, max_norm=1.5)
        # 位置初始化嵌入: 为什么设置成三百
        self.pos_embedding = nn.Embedding(300, self.hidden_size, padding_idx=0, max_norm=1.5)
        # 定义
        # self.gnn = GNN(self.hidden_size)
        self.gnn = GNN(self.hidden_size, 0.5)  # drop0.5
        # mamba相关定义
        self.config = Config(model=Mamba4Rec, config_file_list=['config.yaml'])
        self.mamba = trans_to_cuda(Mamba4Rec(opt, self.config))  # mamba配置

        # Sparse Graph Attention
        self.is_dropout = True
        self.w = 20
        dim = self.hidden_size * 2
        self.dim = dim
        self.LN = nn.LayerNorm(dim)
        self.LN2 = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.activate = F.relu
        self.atten_w0 = nn.Parameter(torch.Tensor(1, dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_bias = nn.Parameter(torch.Tensor(dim))
        self.attention_mlp = nn.Linear(dim, dim)
        self.alpha_w = nn.Linear(dim, 1)
        self.self_atten_w1 = nn.Linear(dim, dim)
        self.self_atten_w2 = nn.Linear(dim, dim)
        self.linear2_1 = nn.Linear(2 * dim, dim, bias=True)

        # 多头注意力参数
        self.num_attention_heads = opt.num_attention_heads
        self.attention_head_size = int(dim / self.num_attention_heads)
        self.multi_alpha_w = nn.Linear(self.attention_head_size, 1)

        # 邻居查找模块
        self.FindNeighbor = FindNeighbors(self.hidden_size,self.nei_n)
        self.w_ne = opt.w_ne
        self.gama = opt.gama

        # 关系图卷积
        self.RelationGraph = RelationGAT(self.batch_size, self.hidden_size)
        self.w_f = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.linear_one = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=True)
        self.linear_two = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=True)
        self.linear_three = nn.Linear(2 * self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.LayerNorm = LayerNorm(2 * self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.2)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    # 模型参数初始化
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    # 添加位置嵌入
    def add_position_embedding(self, sequence):

        batch_size = sequence.shape[0]  # b
        len = sequence.shape[1]  # s

        position_ids = torch.arange(len, dtype=torch.long, device=sequence.device)  # [s,]
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)  # [b,s]
        position_embeddings = self.pos_embedding(position_ids)  # [b,s,d]
        item_embeddings = self.embedding(sequence)

        sequence_emb = torch.cat((item_embeddings, position_embeddings), -1)
        sequence_emb = self.LayerNorm(sequence_emb)

        return sequence_emb

    # 去噪函数
    def denoise(self, alpha):  # [b, s+1, s+1]
        batch_size = alpha.shape[0]
        seq_len = alpha.shape[1]
        alpha_avg = torch.mean(alpha, 2, keepdim=True).expand(batch_size, seq_len,
                                                              seq_len)  # 平均注意力权重 [b,s+1]->[b,s+1,s+1]
        alpha_mask = alpha - 0.1 * alpha_avg

        # 使用阈值过滤，只保留大于0的值，其余置0
        alpha_out = torch.where(alpha_mask > 0, alpha, trans_to_cuda(torch.tensor([0.])))
        return alpha_out

    # 定义一个方法用于增强目标嵌入向量target_emb，基于last_emb（上一次点击的嵌入）的相似度
    def enhanceTarget(self, last_emb, target_emb):  # [b,d],[b,d]

        # 计算嵌入之间的相似度
        def compute_sim(last_emb, target_emb):
            fenzi = torch.matmul(last_emb, target_emb.permute(1, 0))  # 512*512
            fenmu_l1 = torch.sum(last_emb * last_emb + 0.000001, 1)
            fenmu_l2 = torch.sum(target_emb * target_emb + 0.000001, 1)
            fenmu_l1 = torch.sqrt(fenmu_l1).unsqueeze(1)
            fenmu_l2 = torch.sqrt(fenmu_l2).unsqueeze(1)
            fenmu = torch.matmul(fenmu_l1, fenmu_l2.permute(1, 0))
            cos_sim = fenzi / fenmu  # 512*512
            cos_sim = nn.Softmax(dim=-1)(cos_sim)
            return cos_sim  # [b,b]

        # 根据相似度分数调整target_emb
        def compute_pos(batch_size, cos_sim):
            gama = self.gama
            scores = torch.sum(cos_sim, 1)  # [b,]
            value = torch.mean(scores) * gama  # [1,] 相似度得分的均值
            for index in range(batch_size):
                target_emb[index] = torch.where(scores[index] - value > 0,
                                                self.linear2_1(torch.cat([target_emb[index], last_emb[index]], 0)),
                                                target_emb[index])

        # 向量压缩：去除单维度
        target_emb = target_emb.squeeze()  # [b,2d]
        #  调用compute_sim函数计算last_emb和target_emb之间的相似度矩阵
        cos_sim = compute_sim(last_emb, target_emb)  # [b,b]
        batch_size = last_emb.shape[0]  # b
        mask = trans_to_cuda(torch.Tensor(np.diag([1] * batch_size)))  # [b,b] 构造对角矩阵
        scores = cos_sim * mask  # 只有对角线上有值 [b,b] [0.4,0,0,0,0][0,0.5,0,0,0]
        # 这个函数会根据scores更新target_emb
        compute_pos(batch_size, scores)
        up_target = target_emb.unsqueeze(1)  # [b,1,d]
        # 返回经过相似度加权增强后的target_emb
        return up_target

    # 计算注意力权重
    def get_alpha(self, x=None, seq_len=70, number=None):  # x[b,1,d], seq = len为每个会话序列中最后一个元素
        if number == 0:
            alpha_ent = torch.sigmoid(self.alpha_w(x)) + 1  # [b,1,1]
            alpha_ent = self.add_value(alpha_ent).unsqueeze(1)  # [b,1,1]
            alpha_ent = alpha_ent.expand(-1, seq_len, -1)  # [b,s+1,1]
            return alpha_ent
        if number == 1:  # x[b,1,d]
            alpha_global = torch.sigmoid(self.alpha_w(x)) + 1  # [b,1,1]
            alpha_global = self.add_value(alpha_global)
            return alpha_global

    # 计算注意力权重，适用于多头注意力机制
    def get_alpha2(self, x=None, seq_len=70):  # x [b,n,d/n]
        alpha_ent = torch.sigmoid(self.multi_alpha_w(x)) + 1  # [b,n,1]
        alpha_ent = self.add_value(alpha_ent).unsqueeze(2)  # [b,n,1,1]
        alpha_ent = alpha_ent.expand(-1, -1, seq_len, -1)  # [b,n,s,1]
        return alpha_ent

    def add_value(self, value):

        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        return value

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 修改
    def Multi_Self_attention(self, q, k, v, sess_len):
        # q,k,v([512, 40, 200])
        is_dropout = True
        if is_dropout:
            q_ = self.dropout(self.activate(self.attention_mlp(q)))  # [b,s+1,d]
        else:
            q_ = self.activate(self.attention_mlp(q))

        query_layer = self.transpose_for_scores(q_)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)
        # print(query_layer.size())   # ([512, 5, 40, 40])

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        alpha_ent = self.get_alpha2(query_layer[:, :, -1, :], seq_len=sess_len)

        attention_probs = entmax_bisect(attention_scores, alpha_ent, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim,)
        att_v = context_layer.view(*new_context_layer_shape)

        if is_dropout:
            att_v = self.dropout(self.self_atten_w2(self.activate(self.self_atten_w1(att_v)))) + att_v
        else:
            att_v = self.self_atten_w2(self.activate(self.self_atten_w1(att_v))) + att_v

        att_v = self.LN(att_v)
        c = att_v[:, -1, :].unsqueeze(1)  # [b,d]->[b,1,d]
        x_n = att_v[:, :-1, :]  # [b,s,d]
        # print(c.size())    #（512,1,200）
        # print(x_n.size())    #（512,39,200）
        return c, x_n

    # def self_attention(self, q, k, v, mask=None, alpha_ent=1):
    #     is_dropout = True
    #     if is_dropout:
    #         q_ = self.dropout(self.activate(self.attention_mlp(q)))  # [b,s+1,d]
    #     else:
    #         q_ = self.activate(self.attention_mlp(q))
    #     scores = torch.matmul(q_, k.transpose(1, 2)) / math.sqrt(self.dim)  # [b,s+1,d]x[b,d,s+1] = [b,s+1,s+1]
    #     if mask is not None:
    #         mask = mask.unsqueeze(1).expand(-1, q.size(1), -1)
    #         scores = scores.masked_fill(mask == 0, -np.inf)
    #     alpha = entmax_bisect(scores, alpha_ent, dim=-1)  # [b,s+1,s+1] 注意向量
    #     # alpha2 = F.softmax(scores, dim=-1)
    #
    #     att_v = torch.matmul(alpha, v)  #[b,s+1,d]
    #
    #
    #     if is_dropout:
    #         att_v = self.dropout(self.self_atten_w2(self.activate(self.self_atten_w1(att_v)))) + att_v
    #     else:
    #         att_v = self.self_atten_w2(self.activate(self.self_atten_w1(att_v))) + att_v
    #     att_v = self.LN(att_v)
    #     c = att_v[:, -1, :].unsqueeze(1)
    #     x_n = att_v[:, :-1, :]
    #     print(666)
    #     print(c.size())
    #     print(x_n.size())
    #     return c, x_n

    def global_attention(self, target, k, v, mask=None, alpha_ent=1):
        alpha = torch.matmul(torch.relu(k.matmul(self.atten_w1) + target.matmul(self.atten_w2) + self.atten_bias),
                             self.atten_w0.t())
        if mask is not None:  # [b,s]
            mask = mask.unsqueeze(-1)
            alpha = alpha.masked_fill(mask == 0, -np.inf)
        alpha = entmax_bisect(alpha, alpha_ent, dim=1)
        c = torch.matmul(alpha.transpose(1, 2), v)
        return c

    # [b,d], [b,d]
    def decoder(self, global_s, target_s):
        if self.is_dropout:
            c = self.dropout(torch.selu(self.w_f(torch.cat((global_s, target_s), 2))))
        else:
            c = torch.selu(self.w_f(torch.cat((global_s, target_s), 2)))  # [b,1,4d]

        c = c.squeeze()  # [b,d]
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        return l_c

    def compute_scores(self, hidden, mask, target_emb, att_hidden, relation_emb):  # Dual_att[b,d], Dual_g[b,d]
        # ht为local_embedding 取hidden最后一个元素
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        # 使用sigmoid计算加权和
        sess_global = torch.sigmoid(q1 + q2)  # [b,s,d]

        # Sparse Global Attention，计算全局注意力权重
        alpha_global = self.get_alpha(x=target_emb, number=1)  # [b,1,2d]
        # 定义q,k,v计算全局注意力
        q = target_emb
        k = att_hidden  # [b,s,2d]
        v = sess_global  # [b,s,2d]
        # 使用global_attention计算全局注意力的输出
        global_c = self.global_attention(q, k, v, mask=mask, alpha_ent=alpha_global)
        # 使用decoder将全局注意力的输出和target_emb转换为sess_final
        sess_final = self.decoder(global_c, target_emb)
        # SIC：通过findneighbor方法找到邻居并更新sess_final
        neighbor_sess = self.FindNeighbor(sess_final + relation_emb)
        sess_final = sess_final + neighbor_sess

        b = self.embedding.weight[1:] / torch.norm(self.embedding.weight[1:], dim=-1).unsqueeze(1)
        # 计算得分
        scores = self.w * torch.matmul(sess_final, b.transpose(1, 0))  # [b,d]x[d,n] = [b,n]
        return scores

    def forward(self, inputs, A, alias_inputs, A_hat, D_hat):  # inputs[b,s], A[b,s,s]
        # 引入可学习的位置编码𝑋𝑡 = 𝐶𝑜𝑛𝑐𝑎𝑡 (𝑉𝑡 , 𝑃𝑡 )
        seq_emb = self.add_position_embedding(inputs)  # [b,s,2d]
        # gnn层，使用GGNN作为初始编码器
        # hidden = self.gnn(A, seq_emb)  # (b,s,2d)
        hidden = self.gnn(seq_emb, A)  # (b,s,2d)
        # print(hidden.size())
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden_gnn = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])  # [b,s,2d]
        # 创建额外节点𝐻𝑡 = 𝐶𝑜𝑛𝑐𝑎𝑡 (𝐻𝑡 , 𝑡𝑛 ),
        zeros = torch.cuda.FloatTensor(seq_hidden_gnn.shape[0], 1, self.dim).fill_(0)  # [b,1,d]
        # 将节点嵌入和空白节点沿着序列长度维度拼接（将空白节点添加到序列末尾）
        session_target = torch.cat([seq_hidden_gnn, zeros], 1)  # [b,s+1,d]
        # 获得第一维的大小（序列长度）
        sess_len = session_target.shape[1]
        # target_emb = session_target[:, -1, :].unsqueeze(1)  # [b,d]->[b,1,d]
        # x_n = session_target[:, :-1, :]  # [b,s,d]
        target_emb, x_n = self.mamba(session_target, sess_len)
        # # 会话图稀疏自注意力机制
        # target_emb, x_n = self.Multi_Self_attention(session_target, session_target, session_target, sess_len)
        # 关系图稀疏自注意力机制
        # self.RelationGraph = RelationGAT(self.batch_size, self.hidden_size)
        relation_emb = self.RelationGraph(self.embedding.weight, inputs, A_hat, D_hat, target_emb)

        return seq_hidden_gnn, target_emb, x_n, relation_emb


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    # 获得第i个批次的数据切片
    alias_inputs, A, items, mask, targets = data.get_slice(i)  # 得到碎片数据：batch中的值
    # 计算项目重叠部分
    A_hat, D_hat = data.get_overlap(items)

    A_hat = trans_to_cuda(torch.Tensor(A_hat))
    D_hat = trans_to_cuda(torch.Tensor(D_hat))
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    # 调用模型进行前向传播
    hidden, target_emb, att_hidden, relation_emb = model(items, A, alias_inputs, A_hat, D_hat)
    # 调用模型的方法计算得分
    scores = model.compute_scores(hidden, mask, target_emb, att_hidden, relation_emb)

    return targets, scores


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    # 生成训练批次
    slices = train_data.generate_batch(model.batch_size)
    # 遍历每个批次
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        # 前向传播
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        # 计算损失
        loss = model.loss_function(scores, targets - 1)
        # 反向传播
        loss.backward()
        model.optimizer.step()
        # 累计总损失
        total_loss = total_loss + loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()), flush=True)
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    # 生成测试批次
    slices = test_data.generate_batch(model.batch_size)
    # 遍历每个测试批次
    for i in slices:
        # 前向传播
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr



