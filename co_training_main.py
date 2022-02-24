import os
import random

import torch
import numpy as np

from time import time
from tqdm import tqdm
from copy import deepcopy
import logging
from prettytable import PrettyTable

from utils.parser import co_training_parse_args
from utils.data_loader import load_data, Supervised_User_Item_Embeb_Dataset, Unsupervised_User_Item_Embeb_Dataset
from utils.evaluate import test
from utils.helper import early_stopping
import torch.nn as nn
from torch.utils.data import DataLoader
import math


n_users = 0
n_items = 0


def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    ce = nn.CrossEntropyLoss()
    ce.to(device)
    loss1 = ce(logit_S1, labels_S1)
    loss2 = ce(logit_S2, labels_S2) 
    return (loss1+loss2)

def loss_cot(U_p1, U_p2, batch_size):
# the Jensen-Shannon divergence between p1(x) and p2(x)
    S = nn.Softmax(dim = 1)
    LS = nn.LogSoftmax(dim = 1)
    a1 = 0.5 * (S(U_p1) + S(U_p2))
    loss1 = a1 * torch.log(a1)
    loss1 = -torch.sum(loss1)
    loss2 = S(U_p1) * LS(U_p1)
    loss2 = -torch.sum(loss2)
    loss3 = S(U_p2) * LS(U_p2)
    loss3 = -torch.sum(loss3)

    return (loss1 - 0.5 * (loss2 + loss3))/batch_size

def adjust_lamda(epoch, lambda_cot_max):
    global lambda_cot
    if epoch <= args.classifier_warm_up:
        lambda_cot = lambda_cot_max*math.exp(-5*(1-epoch/args.classifier_warm_up)**2)
    else: 
        lambda_cot = lambda_cot_max 


def adjust_learning_rate(optimizer, epoch):
    """cosine scheduling"""
    epoch = epoch + 1
    lr = args.classifier_lr*(1.0 + math.cos((epoch-1)*math.pi/args.classifier_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1):

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
                                                       n_negs*K)).to(device)
    return feed_dict


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = co_training_parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, user_dict, n_params, norm_mat = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K

    """define model"""
    from modules.LightGCN import LightGCN
    from modules.NGCF import NGCF
    from modules.LightGCN import MFBPR

    if args.gnn == 'lightgcn':
        model = LightGCN(n_params, args, norm_mat).to(device)
    else:
        model = NGCF(n_params, args, norm_mat).to(device)
    """define optimizer"""
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.pre_train == 'lightgcn':
        print("use lightgcn for pretrian embedding")
        pretrain_model = model
        pretrain_optimizer = model_optimizer
    else:
        pretrain_model = MFBPR(n_params, args).to(device)
        pretrain_optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=args.lr)
    

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    print("start training ...")
    if args.load_pretrain_model == False:
        print("start pre-training lightgcn Embedding for classifier")
        print("Pre-training lightgcn using the original sampling setting")
        pretrain_model.ns = 'rns'
        for epoch in tqdm(range(args.pretrain_epoch)):
            train_cf_ = train_cf
            index = np.arange(len(train_cf_))
            np.random.shuffle(index)
            train_cf_ = train_cf_[index].to(device)

            """training"""
            pretrain_model.train()
            loss, s = 0, 0
            hits = 0
            train_s_t = time()
            while s + args.batch_size <= len(train_cf):
                batch = get_feed_dict(train_cf_,
                                    user_dict['train_user_set'],
                                    s, s + args.batch_size,
                                    n_negs)

                batch_loss, _, _ = pretrain_model(batch=batch)

                pretrain_optimizer.zero_grad()
                batch_loss.backward()
                pretrain_optimizer.step()

                loss += batch_loss
                s += args.batch_size
        torch.save(pretrain_model.state_dict(), args.save_address_pretrain_model)
    else:
        print("load pre-training lightgcn Embedding for classifier")
        pretrain_model.load_state_dict(torch.load(args.save_address_pretrain_model))
        model = pretrain_model
        


    print("start training the co-training Classifier")
    pretrain_model.eval()
    from modules.Classifier import Classifier
    supervised_dataset = Supervised_User_Item_Embeb_Dataset(train_cf=train_cf,train_user_set=user_dict['train_user_set'],pretrained_model=model, neg_k=2, n_negs=4, batch_size=2048, use_light_gcn=True, device=device)
    supervised_dataset_2 = Supervised_User_Item_Embeb_Dataset(train_cf=train_cf,train_user_set=user_dict['train_user_set'],pretrained_model=model, neg_k=2, n_negs=4, batch_size=2048, use_light_gcn=True, device=device, rand_seed=1010)
    unsupervised_dataset = Unsupervised_User_Item_Embeb_Dataset(train_cf=train_cf,train_user_set=user_dict['train_user_set'],pretrained_model=model, neg_k=2, n_negs=4, batch_size=2048, use_light_gcn=True, device=device)
    supervised_dataloader = DataLoader(dataset=supervised_dataset,batch_size=1024)
    supervised_dataloader_2 = DataLoader(dataset=supervised_dataset_2,batch_size=1024)
    unsupervised_dataloader = DataLoader(dataset=unsupervised_dataset,batch_size=1024)
    
    lambda_cot_max = args.classifier_lambda_cot_max

    net1 = Classifier(input_dim=128,number_label=2)
    net2 = Classifier(input_dim=128,number_label=2)


    net1.to(device)
    net2.to(device)
    net1.train()
    net2.train()

    params = list(net1.parameters()) + list(net2.parameters())
    classifier_optimizer = torch.optim.SGD(params, lr=args.classifier_lr, momentum=args.classifier_momentum, weight_decay=args.classifier_decay)

    for epoch in tqdm(range(args.classifier_epochs)):

      adjust_learning_rate(classifier_optimizer,epoch)
      adjust_lamda(epoch,lambda_cot_max)

      U_dataloader_iterator = iter(unsupervised_dataloader)
      S2_dataloader_iterator = iter(supervised_dataloader_2)
      
      for i,dt in enumerate(supervised_dataloader):
        inputs_S = dt[0].to(device)
        labels_S = dt[1].to(device)
        labels_S = torch.squeeze(labels_S.long())

        try:
            dt2 = S2_dataloader_iterator.next()
            inputs_S2 = dt2[0].to(device)
            labels_S2 = dt2[1].to(device)
            labels_S2 = torch.squeeze(labels_S2.long())
        except StopIteration:
            S2_dataloader_iterator = iter(supervised_dataloader_2)
            dt2 = S2_dataloader_iterator.next()
            inputs_S2 = dt2[0].to(device)
            labels_S2 = dt2[1].to(device)
            labels_S2 = torch.squeeze(labels_S2.long())
        
        # Tackling the different data size between two dataloaders
        try:
            inputs_U = U_dataloader_iterator.next().to(device)
        except StopIteration:
            U_dataloader_iterator = iter(unsupervised_dataloader)
            inputs_U = U_dataloader_iterator.next().to(device)


        # inputs_S, labels_S = inputs_S.cuda(), labels_S.cuda()
        # inputs_U = inputs_U.cuda()   

        logit_S1 = net1(inputs_S)
        logit_S2 = net2(inputs_S2)
        logit_U1 = net1(inputs_U)
        logit_U2 = net2(inputs_U)

        classifier_optimizer.zero_grad()
        net1.zero_grad()
        net2.zero_grad()

        Loss_sup = loss_sup(logit_S1, logit_S2, labels_S, labels_S2)
        Loss_cot = loss_cot(logit_U1, logit_U2, batch_size=1024)

        total_loss = Loss_sup + lambda_cot*Loss_cot
        total_loss.backward()
        classifier_optimizer.step()
    
    print("Finish training the co-training Classifier")

    net1.eval()
    net2.eval()


    print("Start training LightGCN with co-training")

    model.ns = 'co'
    for epoch in tqdm(range(args.epoch)):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)

            batch_loss, _, _ = model(net1=net1, net2=net2, batch=batch)

            model_optimizer.zero_grad()
            batch_loss.backward()
            model_optimizer.step()

            loss += batch_loss
            s += args.batch_size

        train_e_t = time()

        if epoch % 10 == 0:
            """testing"""
            print("testing")
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
            with torch.no_grad():
                model.eval()
                test_s_t = time()
                test_ret = test(model, user_dict, n_params, mode='test')
                test_e_t = time()
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret['ndcg'],
                    test_ret['precision'], test_ret['hit_ratio']])

                if user_dict['valid_user_set'] is None:
                    valid_ret = test_ret
                else:
                    test_s_t = time()
                    valid_ret = test(model, user_dict, n_params, mode='valid')
                    test_e_t = time()
                    train_res.add_row(
                        [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'], valid_ret['ndcg'],
                        valid_ret['precision'], valid_ret['hit_ratio']])
                

                # *********************************************************
                # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
                cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][0], cur_best_pre_0,stopping_step, expected_order='acc',flag_step=10)
                print(train_res.get_string())  
                with open(args.save_running_log+"_"+args.gnn+"_"+args.dataset+"_"+str(args.classifier_lr)+"_"+str(args.classifier_decay)+"_"+str(args.classifier_lambda_cot_max)+"_cotrain.txt", 'a') as f:
                    f.write(train_res.get_string())       

                if should_stop:
                    break

                """save weight"""
                if valid_ret['recall'][0] == cur_best_pre_0 and args.save:
                    torch.save(model.state_dict(), args.out_dir+args.gnn+"_"+args.dataset+"_"+ 'model_' + '.ckpt')
        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
