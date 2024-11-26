#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""
# 导入NetworkX库，用于创建和操作图结构。
import networkx as nx
import numpy as np


# 定义一个函数，用于生成数据掩码和填充数据以匹配最长序列长度
def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    # 将每个序列填充到最大长度，使用item_tail作为填充值
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    # 生成掩码数组，1表示实际数据，0表示填充数据。
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


# 定义一个函数，用于将训练集分割为训练集和验证集。
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    # 关联图（计算全局图边权重：共同点击数/全部点击数
    def get_overlap(self, sessions):
        # 初始化一个与会话列表长度相同的零矩阵，用于存储会话之间的重叠度
        matrix = np.zeros((len(sessions), len(sessions)))
        # 遍历会话列表
        for i in range(len(sessions)):
            seq_a = set(sessions[i])   # 将会话转成集合
            seq_a.discard(0)           # 移除元素0
            # 从当前会话的下一个会话开始遍历
            for j in range(i+1, len(sessions)):
                # 相同操作
                seq_b = set(sessions[j])
                seq_b.discard(0)

                # 计算两个会话的交集
                overlap = seq_a.intersection(seq_b)
                # 计算两个集合的并集
                ab_set = seq_a | seq_b
                # 计算重叠度
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        # 在矩阵对角线上添加 1.0，表示每个会话与自身的重叠度为 1
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)     # 计算总重叠度
        degree = np.diag(1.0/degree)             # 计算每个会话度的倒数
        return matrix, degree

    # 生成数据批次
    def generate_batch(self, batch_size):
        if self.shuffle:
            # 生成一个索引数组，从0到长度减一
            shuffled_arg = np.arange(self.length)
            # 打乱索引数组
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        # 如果数据长度除以批次有余数，增加一个批次容纳剩余数据
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices


    # 根据索引i获取特定批次的数据
    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        # 计算每个批次 每个序列的节点数量（inputs是一个批次的所有序列）
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        # 计算所有序列的最大节点数量
        max_n_node = np.max(n_node)
        # 构建邻接矩阵
        for u_input in inputs:
            node = np.unique(u_input)   # 获取唯一节点
            # 将节点转化为列表，用0填充到最大节点数量
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            # 初始化邻接矩阵
            u_A = np.zeros((max_n_node, max_n_node))
            # 遍历序列中的元素对
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:    # 0为序列结束的标记
                    break
                # 找到序列中连续元素在节点列表中对应的索引
                # np.where(node == u_input[i])是个数组，包含所有满足条件的元素
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)   # 计算入度
            u_sum_in[np.where(u_sum_in == 0)] = 1   # 将入度为0的节点设置为1
            u_A_in = np.divide(u_A, u_sum_in)    # 矩阵除法（归一化）
            u_sum_out = np.sum(u_A, 1)  # 计算出度
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            # u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            u_A = u_A_in + u_A_out
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets
