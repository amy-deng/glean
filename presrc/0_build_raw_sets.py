import glob
import sys
import numpy as np
import pandas as pd
import os
import time
import argparse
print(os.getcwd())

# build train/valid/test sets by time
# using full_events(with event sentence) the dates are fixed

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def split(args):
    num_nodes, num_rels = get_total_number(args.dp + args.dn, 'stat.txt')

    quadruple_idx_path = args.dp + args.dn + '/quadruple_idx.txt'
    df = pd.read_csv(quadruple_idx_path, sep='\t',  lineterminator='\n', names=[
                     'source', 'relation', 'target', 'time'])
    print(df.head())
    # ratio 80% 10% 10%
    # in total 2557 days (0-2556) 
    cut_1 = 1795 #2044
    cut_2 = 2019 #2300
    train_df = df.loc[df['time'] <= cut_1]
    valid_df = df.loc[(df['time'] > cut_1) & (df['time'] <= cut_2)]
    test_df = df.loc[df['time'] > cut_2]

    train_path = args.dp + args.dn + '/train.txt'
    valid_path = args.dp + args.dn + '/valid.txt'
    test_path = args.dp + args.dn + '/test.txt'
    train_df.to_csv(train_path, sep='\t', encoding='utf-8', header=None, index=False)
    valid_df.to_csv(valid_path, sep='\t', encoding='utf-8', header=None, index=False)
    test_df.to_csv(test_path, sep='\t', encoding='utf-8', header=None, index=False)
    print(train_path, 'saved')
    print(valid_path, 'saved')
    print(test_path, 'saved')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--dp", default="../data/", help="dataset path")
    ap.add_argument("--dn", default="AFG", help="dataset name (will create a folder)")

    args = ap.parse_args()
    print(args)

    if not os.path.exists("{}{}".format(args.dp, args.dn)):
        print('Check if the dataset exists')
        exit()

    split(args)
