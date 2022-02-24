import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
import torch
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import random

n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)


def read_cf_amazon(file_name):
    return np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]


def read_cf_yelp2018(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)


def statistics(train_data, valid_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1

    if dataset != 'yelp2018':
        n_items -= n_users
        # remap [n_users, n_users+n_items] to [0, n_items]
        train_data[:, 1] -= n_users
        valid_data[:, 1] -= n_users
        test_data[:, 1] -= n_users

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))


def build_sparse_graph(data_cf):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_users+n_items)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items))
    return _bi_norm_lap(mat)


def load_data(model_args):
    global args, dataset
    args = model_args
    dataset = args.dataset
    directory = args.data_path + dataset + '/'

    if dataset == 'yelp2018':
        read_cf = read_cf_yelp2018
    else:
        read_cf = read_cf_amazon

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    if args.dataset != 'yelp2018':
        valid_cf = read_cf(directory + 'valid.txt')
    else:
        valid_cf = test_cf
    statistics(train_cf, valid_cf, test_cf)

    print('building the adj mat ...')
    norm_mat = build_sparse_graph(train_cf)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
    }
    user_dict = {
        'train_user_set': train_user_set,
        'valid_user_set': valid_user_set if args.dataset != 'yelp2018' else None,
        'test_user_set': test_user_set,
    }

    print('loading over ...')
    return train_cf, user_dict, n_params, norm_mat


class Supervised_User_Item_Embeb_Dataset(Dataset):
    def __init__(self, train_cf, train_user_set, pretrained_model, neg_k,n_negs,batch_size,use_light_gcn = True, device=torch.device("cuda:1"), rand_seed = 2020):
        
        print(f"using rand_seed{rand_seed} for supervised_ui_dataset")
        
        with torch.no_grad():

          self.data = None
          self.label = None
          self.device = device
          s = 0

          if use_light_gcn == True:
            user_gcn_emb, item_gcn_emb = pretrained_model.generate(split=True)

          while s + batch_size <= len(train_cf):
            #   print(s)
              batch = self.get_feed_dict(train_cf, train_user_set, s, s + batch_size, n_negs, rand_seed)
              pretrained_model.eval()

              if use_light_gcn == True:
              
                user, pos_i, neg_i = pretrained_model.get_user_item_embedding(user_gcn_emb, item_gcn_emb ,batch, True)
              else: 
                user, pos_i, neg_i = pretrained_model.get_user_item_embedding(batch,neg_k, True)

              pos_cat = torch.cat((user,pos_i),dim=1)
              pos_label = torch.ones(pos_cat.shape[0], 1)

              neg_cat = torch.cat((user,neg_i),dim=1)
              neg_label = torch.zeros(neg_cat.shape[0], 1)

              # dim = user.shape[1]

              # neg_cat = None
              # for i, u in enumerate(user):
                
              #   ui_stack = torch.cat((user[i].expand(neg_i.shape[1],dim),neg_i[i]),dim=1)
              #   if i == 0:
              #     neg_cat = ui_stack
              #   else:
              #     neg_cat = torch.cat((neg_cat,ui_stack),dim=0)

              # neg_label = torch.zeros(neg_cat.shape[0],1)

              if s == 0:
                self.data = pos_cat
                self.label = pos_label
                self.data = torch.cat((self.data, neg_cat),dim=0)
                self.label = torch.cat((self.label, neg_label),dim=0)
              else:
                self.data = torch.cat((self.data, pos_cat),dim=0)
                self.label = torch.cat((self.label, pos_label),dim=0)
                self.data = torch.cat((self.data, neg_cat),dim=0)
                self.label = torch.cat((self.label, neg_label),dim=0)
              s += batch_size
        
        seed = 2020
        random.seed(seed)
            
    def get_feed_dict(self,train_entity_pairs, train_pos_set, start, end, n_negs=1, seed = 2020):

        random.seed(seed)
        def sampling(user_item, train_set, n):
            neg_items = []
            for user, _ in user_item.cpu().numpy():
                user = int(user)
                negitems = []
                for i in range(n):  # sample n times
                    while True:
                        negitem = random.choice(range(n_items))
                        if negitem not in train_set[user]:
                            break
                    negitems.append(negitem)
                neg_items.append(negitems)
            return neg_items

        feed_dict = {}
        entity_pairs = train_entity_pairs[start:end]
        feed_dict['users'] = entity_pairs[:, 0]
        feed_dict['pos_items'] = entity_pairs[:, 1]
        feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                          train_pos_set,
                                                          n_negs)).to(self.device)
        return feed_dict

    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return (data,label)

class Unsupervised_User_Item_Embeb_Dataset(Dataset):
    def __init__(self, train_cf, train_user_set, pretrained_model, neg_k,n_negs,batch_size, use_light_gcn = True, device=torch.device("cuda:1")):

        with torch.no_grad():

          self.data = None
          self.label = None
          self.device = device
          s = 0
          if use_light_gcn == True:
            user_gcn_emb, item_gcn_emb = pretrained_model.generate(split=True)

          while s + batch_size <= len(train_cf):
            #   print(s)
              batch = self.get_feed_dict(train_cf, train_user_set, s, s + batch_size, n_negs)
              pretrained_model.eval()
              if use_light_gcn == True:
                user, neg_i = pretrained_model.get_user_item_embedding(user_gcn_emb, item_gcn_emb,batch, False)
              else:
                user, neg_i = pretrained_model.get_user_item_embedding(batch,neg_k, False)
              # pos_cat = torch.cat((user,pos_i),dim=1)
              # pos_label = torch.ones(pos_cat.shape[0], 1)
              dim = user.shape[1]
              neg_cat = None
              for i, u in enumerate(user):
                
                ui_stack = torch.cat((user[i].expand(neg_i.shape[1],dim),neg_i[i]),dim=1)
                if i == 0:
                  neg_cat = ui_stack
                else:
                  neg_cat = torch.cat((neg_cat,ui_stack),dim=0)

              # neg_label = torch.zeros(neg_cat.shape[0],1)

              if s == 0:
                self.data = neg_cat
              else:
                self.data = torch.cat((self.data, neg_cat),dim=0)
              s += batch_size
            
    def get_feed_dict(self,train_entity_pairs, train_pos_set, start, end, n_negs=1):

        def sampling(user_item, train_set, n):
            neg_items = []
            for user, _ in user_item.cpu().numpy():
                user = int(user)
                negitems = []
                for i in range(n):  # sample n times
                    while True:
                        negitem = random.choice(range(n_items))
                        if negitem not in train_set[user]:
                            break
                    negitems.append(negitem)
                neg_items.append(negitems)
            return neg_items

        feed_dict = {}
        entity_pairs = train_entity_pairs[start:end]
        feed_dict['users'] = entity_pairs[:, 0]
        feed_dict['pos_items'] = entity_pairs[:, 1]
        feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                          train_pos_set,
                                                          n_negs)).to(self.device)
        return feed_dict

    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        data = self.data[idx]
        return data

