# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:02:54 2022

@author: user
"""

import torch
import torch_geometric
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from  torch_geometric.data import DataLoader

class PyGToyDataset(InMemoryDataset):
    def __init__(self, save_root, transform=None, pre_transform=None):
        """
        :param save_root:保存數據的目錄
        :param pre_transform:在讀数据前的預處理操作
        :param transform:在访问之前动态转换数据对象(因此最好用于数据扩充)
        """
        super(PyGToyDataset, self).__init__(save_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):  # 原始數據存放位置
        return ['origin_dataset']

    @property
    def processed_file_names(self):
        return ['toy_dataset3.pt']

    def download(self):  # 本次使用自己的數據，而不是從網路上抓，因此pass掉
        pass

    def process(self):   # 處理數據的函數（如何創建跟保存）
        myArch = np.load('D:/material_project/cif_to_graph/npz/conventional_standard/gcn_data/train_2.npz',allow_pickle = True)
        np_data = myArch['graph_dict'].item()
        Id = np_data.keys()
        n = 0
        data_list = []
        for n in Id:
            data = Data(x = torch.Tensor(np_data[n]['node']), edge_index = np_data[n]['edge_index'], y = torch.FloatTensor([np_data[n]['volume']]))
            data_list.append(data)
        #data_list = [toy_dataset(num_nodes=32, num_node_features=3, num_edges=42) for _ in range(100)]
        data_save, data_slices = self.collate(data_list) # 直接保存list可能很慢，所以使用collate函數轉成大的torch_geometric.data.Data
        torch.save((data_save, data_slices), self.processed_paths[0])


if __name__ == "__main__":
    # toy_sample = toy_dataset(num_nodes=32, num_node_features=3, num_edges=42)
    # print(toy_sample)
    toy_data = PyGToyDataset(save_root="D:/material_project/cif_to_graph/gcn_data")  
    # print(toy_data[0])
    data_loader = DataLoader(toy_data, batch_size=5, shuffle=True) # batch_size=5實現平行化(5张图放一起)

    for batch in data_loader:
        print(batch.edge_index)
