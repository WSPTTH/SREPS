import os
import json
import argparse
from preprocess import preprocess
from dataset import Dataset, DatasetSpliter
from model import SREPS


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--mode', type=str, default='', help='support prepro/run/eval')

    # for `prepro` to handle the four dataset.
    parser.add_argument('--dataset', type=str, default='filmtrust',
                        help='the dataset name, support ciao/epinions/filmtrust/flixster')
    parser.add_argument('--data_input', type=str, default='', help='the input folder path')
    parser.add_argument('--data_output', type=str, default='data/', help='the output folder path')

    # for dataset (split and save)
    parser.add_argument('--split_ratio', type=float, default=0.2, help='the ratio of the dev dataset')
    parser.add_argument('--train_file', type=str, default='data/filmtrust/train.data', help='the path of the train file')
    parser.add_argument('--dev_file', type=str, default='data/filmtrust/dev.data', help='the path of the dev file')
    parser.add_argument('--rating_file', type=str, default='data/filmtrust/ratings.total', help='rating file')
    parser.add_argument('--link_file', type=str, default='data/filmtrust/links.total', help='link file')

    # for model
    parser.add_argument('--dim', type=int, default=5, help='the dimension of the rating semantic space')
    parser.add_argument('--udim', type=int, default=5, help='the dimension of the essential preference space')
    parser.add_argument('--ldim', type=int, default=5, help='the dimension of the social semantic space')
    parser.add_argument('--neg_num', type=int, default=5, help='negative sample number')
    parser.add_argument('--alpha', type=float, default=0.2, help='the loss weight for the social network')
    parser.add_argument('--beta', type=float, default=0.1, help='the loss weight for the recommendation network')
    parser.add_argument('--lr', type=float, default=0.005, help='the learning rate')
    parser.add_argument('--lam', type=float, default=0.01, help='the regularization parameter')
    parser.add_argument('--batch', type=int, default=128, help='the batch number')
    parser.add_argument('--iter_num', type=int, default=600000, help='the iteration number')
    parser.add_argument('--save_step', type=int, default=20000, help='the step to save model')
    parser.add_argument('--save_path', type=str, default='model/model.npz', help='the path to save model')

    args = parser.parse_args()

    if args.mode == 'prepro':
        preprocess(args.dataset, args.data_input, args.data_output)
    elif args.mode == 'run':
        if not os.path.exists(args.train_file) or not os.path.exists(args.dev_file):
            train_data, dev_data = DatasetSpliter(args.split_ratio).split(args.rating_file, args.link_file)
            train_data.save(args.train_file)
            dev_data.save(args.dev_file)
        else:
            train_data = Dataset().load(args.train_file)
            dev_data = Dataset().load(args.dev_file)
        model = SREPS(train_data, dim=args.dim, ldim=args.ldim, udim=args.udim, neg_num=args.neg_num,
                      alpha=args.alpha, beta=args.beta, lr=args.lr, batch=args.batch, lam=args.lam,
                      iter_num=args.iter_num, save_path=args.save_path, save_step=args.save_step)
        model.train({'dev': dev_data, 'train': train_data})
    elif args.mode == 'eval':
        dev_data = Dataset().load(args.dev_file)
        model = SREPS(dev_data).load(args.save_path)
        res = model.eval([dev_data])
        print(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True))
