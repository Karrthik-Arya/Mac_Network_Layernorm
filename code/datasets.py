from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import glob
from pathlib import Path
import json
import h5py
import re
import random

from config import cfg


class ClevrDataset(data.Dataset):
    def __init__(self, data_dir, split='train'):

        with open(os.path.join(data_dir, '{}.pkl'.format(split)), 'rb') as f:
            self.data = pickle.load(f)

        # with open(os.path.join(data_dir, 'dic.pkl'), 'rb') as f:
        #     dic = pickle.load(f)
        # ans_dic = dic["answer_dic"]
        # print("classes: ", max(ans_dic.values()))
        self.img = h5py.File(os.path.join(data_dir, '{}.h5'.format(split)), 'r')['features']

    def __getitem__(self, index):
        imgfile, question, answer = self.data[index]
        img = torch.from_numpy(self.img[imgfile])

        return img, question, len(question), answer, "source", imgfile

    def __len__(self):
        return len(self.data)
    
class TestDataset(data.Dataset):
    def __init__(self, data_dir, split = 'train'):
        with open(os.path.join(data_dir, 'targ_{}.pkl'.format(split)), 'rb') as f:
            self.data = pickle.load(f)
        
        with open(os.path.join(data_dir, 'targ_img_map.pkl'.format(split)), 'rb') as f:
            self.img_map = pickle.load(f)

        self.img = h5py.File(os.path.join(data_dir, 'targ_images.h5'), 'r')['features']

    def __getitem__(self, index):
        imgfile, question, answer = self.data[index]
        img_idx = self.img_map[imgfile]
        img = torch.from_numpy(self.img[img_idx])

        return img, question, len(question), answer, "target", imgfile
    
    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    images, lengths, answers, domains, imgfiles,  _ = [], [], [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, domain, imgfile = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        domains.append(domain)
        imgfiles.append(imgfile)

    return {'image': torch.stack(images), 'question': torch.from_numpy(questions),
            'answer': torch.LongTensor(answers), 'question_length': lengths, "domain": domains, "imgfile": imgfiles}
