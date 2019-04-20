import os
import re
from tqdm import tqdm
from collections import namedtuple


def read_rating(in_file, out_file, r_pos=2):
    """write the rating file into the same format"""
    o_base = os.path.split(out_file)[0]
    if not os.path.exists(o_base):
        os.makedirs(o_base)

    with open(in_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc='Read ratings from `{}`'.format(in_file), ascii=True):
            line = re.split('\s+', line.strip())
            user, item, rating = line[0], line[1], line[r_pos]
            fout.write('{} {} {}\n'.format(user, item, rating))


def read_links(in_file, out_file, is_direction_edge=False):
    o_base = os.path.split(out_file)[0]
    if not os.path.exists(o_base):
        os.makedirs(o_base)

    with open(in_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc='Read links from `{}`'.format(in_file), ascii=True):
            line = re.split('\s+', line.strip())
            u0, u1 = line[0], line[1]
            fout.write('{} {}\n'.format(u0, u1))
            if not is_direction_edge:
                fout.write('{} {}\n'.format(u1, u0))


_DataBase = namedtuple('_DataBase', ('rin', 'lin', 'r_pos', 'is_dict'))
_DatasetSetting = {
    'ciao': _DataBase('ratings.txt', 'trust.txt', 3, True),
    'epinions': _DataBase('ratings.txt', 'trust.txt', 3, True),
    'filmtrust': _DataBase('ratings.txt', 'trust.txt', 2, True),
    'flixster': _DataBase('ratings.txt', 'links.txt', 2, True),
}


def preprocess(name, in_base, out_base):
    # transform the four dataset into same format
    data_base = _DatasetSetting[name]
    read_rating(os.path.join(in_base, data_base.rin), os.path.join(out_base, name, 'ratings.total'))
    read_links(os.path.join(in_base, data_base.lin), os.path.join(out_base, name, 'links.total'))

