def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import numpy as np
import time
import utils
import os
from sklearn.utils import shuffle
from models import *
from data import *
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

parser = argparse.ArgumentParser(description='Utilize global information of word and entity graphs')
parser.add_argument("--dp", type=str, default="../data/", help="data path")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
parser.add_argument("--n-hidden", type=int, default=100, help="number of hidden units")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
parser.add_argument("-d", "--dataset", type=str, default='AFG', help="dataset to use")
parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
parser.add_argument("--max-epochs", type=int, default=10, help="maximum epochs")
parser.add_argument("--seq-len", type=int, default=7)
parser.add_argument("--num-r", type=int, default=20, help="number of rel to consider")
parser.add_argument("-b","--batch-size", type=int, default=1)
parser.add_argument("--rnn-layers", type=int, default=1)
parser.add_argument("--maxpool", type=int, default=1)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--use-gru", type=int, default=1, help='1 use gru 0 rnn')
parser.add_argument("--attn", type=str, default='', help='dot/add/genera; default general')
parser.add_argument("--seed", type=int, default=42, help='random seed')
parser.add_argument("--runs", type=int, default=5, help='number of runs')

args = parser.parse_args()
print(args)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
use_cuda = args.gpu >= 0 and torch.cuda.is_available()
print("cuda",use_cuda)
np.random.seed(args.seed)
torch.manual_seed(args.seed) 
# eval metrics
hloss_list = []
hits_list = []
mrr_list = []
mr_list = []
iterations = 0 
while iterations < args.runs:
    iterations += 1
    print('-------------- iterations ',iterations)
    
    if iterations == 1:
        # loading data...
        num_nodes, num_rels = utils.get_total_number(
            args.dp + args.dataset, 'stat.txt')

        with open('{}{}/100.w_emb'.format(args.dp, args.dataset), 'rb') as f:
            word_embeds = pickle.load(f,encoding="latin1")
        word_embeds = torch.FloatTensor(word_embeds)
        vocab_size = word_embeds.size(0)

        train_dataset_loader = EntDistGivenTRData(args.dp, args.dataset, num_nodes, num_rels, 'train', args.seq_len, num_r=args.num_r)
        valid_dataset_loader = EntDistGivenTRData(args.dp, args.dataset, num_nodes, num_rels, 'valid', args.seq_len, num_r=args.num_r)
        test_dataset_loader = EntDistGivenTRData(args.dp, args.dataset, num_nodes, num_rels, 'test', args.seq_len, num_r=args.num_r)
        train_loader = DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True, collate_fn=collate_6)
        valid_loader = DataLoader(valid_dataset_loader, batch_size=1, shuffle=False, collate_fn=collate_6)
        test_loader = DataLoader(test_dataset_loader, batch_size=1, shuffle=False, collate_fn=collate_6)

        try:
            with open(args.dp  + args.dataset+'/wg_r_dict_sl{}_rand_{}.txt'.format(args.seq_len, args.num_r), 'rb') as f:
                word_graph_dict = pickle.load(f)
            with open(args.dp  + args.dataset+'/dg_r_dict_sl{}_rand_{}.txt'.format(args.seq_len, args.num_r), 'rb') as f:
                graph_dict = pickle.load(f)
        except:
            print('Cannot find the dataset')
            exit()
             
        # load word_relation_map.txt
        with open(args.dp  + args.dataset+'/word_relation_map.txt', 'rb') as f:
            rel_map = pickle.load(f)
        # load word_entity_map.txt
        with open(args.dp  + args.dataset+'/word_entity_map.txt', 'rb') as f:
            ent_map = pickle.load(f)
    
    model = glean_actor(h_dim=args.n_hidden, num_ents=num_nodes,   
                                num_rels=num_rels, dropout=args.dropout, 
                                seq_len=args.seq_len,
                                maxpool=args.maxpool,
                                use_gru=args.use_gru,
                                attn=args.attn)

    model_name = model.__class__.__name__
    token = '{}_sl{}_max{}_gru{}_attn{}'.format(model_name, args.seq_len, int(args.maxpool), int(args.use_gru),str(args.attn))
    
    print('Token:', token, args.dataset)

    optimizer = torch.optim.Adam( # SGD
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#params:', total_params)

    os.makedirs('models', exist_ok=True)
    os.makedirs('models/' + args.dataset, exist_ok=True)
    model_state_file = 'models/{}/{}.pth'.format(args.dataset, token)
    model_graph_file = 'models/{}/{}_graph.pth'.format(args.dataset, token)
    outf = 'models/{}/{}.result'.format(args.dataset, token)
    if use_cuda:
        model.cuda()
        word_embeds = word_embeds.cuda()

    model.word_embeds = word_embeds
    model.graph_dict = graph_dict
    model.word_graph_dict = word_graph_dict
    model.ent_map = ent_map
    model.rel_map = rel_map


    @torch.no_grad()
    def evaluate(data_loader, dataset_loader, set_name='valid'):
        model.eval()
        true_rank_l = []
        prob_rank_l = []
        total_ranks = np.array([])
        total_loss = 0
        for i, batch in enumerate(tqdm(data_loader)):
            t_data, r_data, r_hist, r_hist_t, true_s, true_o = batch
            t_data = torch.stack(t_data, dim=0)
            r_data = torch.stack(r_data, dim=0)
            true_s = torch.stack(true_s, dim=0)
            true_o = torch.stack(true_o, dim=0)
            true_rank_s, prob_rank_s, true_rank_o, prob_rank_o, ranks, loss = model.evaluate(t_data, r_data, r_hist, r_hist_t, true_s, true_o)
            total_loss += loss.item()
            true_rank_l.append(true_rank_s.cpu().tolist())
            prob_rank_l.append(prob_rank_s.cpu().tolist())
            true_rank_l.append(true_rank_o.cpu().tolist())
            prob_rank_l.append(prob_rank_o.cpu().tolist())
            total_ranks = np.concatenate((ranks.cpu().numpy(), total_ranks))
    
        print('{} results'.format(set_name)) 
        hloss, recall, f1, f2 = utils.print_eval_metrics(true_rank_l,prob_rank_l,prt=False)
        hits, mrr, mr = utils.print_hit_eval_metrics(total_ranks)
        reduced_loss = total_loss / (dataset_loader.len / 1.0)
        print("{} Loss: {:.6f}".format(set_name, reduced_loss))
        return hloss, recall, f1, f2, hits, mrr, mr

    def train(data_loader, dataset_loader):
        model.train()
        total_loss = 0
        t0 = time.time()
        for i, batch in enumerate(tqdm(data_loader)):
            t_data, r_data, r_hist, r_hist_t, true_s, true_o = batch
            t_data = torch.stack(t_data, dim=0)
            r_data = torch.stack(r_data, dim=0)
            true_s = torch.stack(true_s, dim=0)
            true_o = torch.stack(true_o, dim=0)

            loss_s = model(t_data, r_data, r_hist, r_hist_t, true_s, true_o, False)
            loss_o = model(t_data, r_data, r_hist, r_hist_t, true_s, true_o, True)
            loss = loss_s + loss_o
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_norm)   
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        t2 = time.time()
        reduced_loss = total_loss / (dataset_loader.len / args.batch_size)
        print("Epoch {:04d} | Loss {:.6f} | time {:.2f} {}".format(
            epoch, reduced_loss, t2 - t0, time.ctime()))
        return reduced_loss

 

    bad_counter = 0
    loss_small =  float("inf")
    try:
        # print("start training...")
        for epoch in range(1, args.max_epochs+1):
            train_loss = train(train_loader, train_dataset_loader)
            # evaluate(train_eval_loader, train_dataset_loader, set_name='Train') # eval on train set
            valid_loss, recall, f1, f2, hits, mrr, mr = evaluate(
                valid_loader, valid_dataset_loader, set_name='Valid') # eval on valid set

            if valid_loss < loss_small:
                loss_small = valid_loss
                bad_counter = 0
                print('save better model... #params:', total_params) 
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'global_emb': None}, model_state_file)
                # evaluate(test_loader, test_dataset_loader, set_name='Test')
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break


    except KeyboardInterrupt:
        print('-' * 80)
        print('Exiting from training early, epoch', epoch)

    # Load the best saved model.
    print("\nstart testing...")
    checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    hloss, recall, f1, f2, hits, mrr, mr = evaluate(test_loader, test_dataset_loader, set_name='Test')
    print(args)
    print(token, args.dataset)
    if mrr < 0.1:
        continue
    hloss_list.append(hloss)
    hits_list.append(hits)
    mrr_list.append(mrr)
    mr_list.append(mr)


print('finish training, results ....')
# save average results
hloss_list = np.array(hloss_list)
hits_list = np.array(hits_list)
mrr_list = np.array(mrr_list)
mr_list = np.array(mr_list)

hloss_avg, hloss_std = hloss_list.mean(0), hloss_list.std(0)
hits_avg, hits_std = hits_list.mean(0), hits_list.std(0)
mrr_avg, mrr_std = mrr_list.mean(0), mrr_list.std(0)
mr_avg, mr_std = mr_list.mean(0), mr_list.std(0)
print('--------------------')
print("hamming loss: {:.4f}".format(hloss_avg))
print("Hits @1: {:.4f} | @3: {:.4f} | @10: {:.4f}  ".format(hits_avg[0],hits_avg[1],hits_avg[2]))
# print("MRR: {:.4f} | MR: {:.4f}".format(mrr_avg,mr_avg))
# save it !!! 
# all_results = [
#     hloss_list, hits_list,
#     [hloss_avg, hloss_std],
#     [mrr_avg, mrr_std],
#     [mr_avg, mr_std]
# ]
# with open(outf,'wb') as f:
#     pickle.dump(all_results, f)