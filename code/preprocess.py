import os
import sys
import json
import pickle

import nltk
import tqdm
from PIL import Image
import pandas as pd

def load_answer_dict(file_path):
    ans_dict = {}
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            word = line.strip()  
            ans_dict[word] = idx  
    return ans_dict

def process_question(root, split, word_dic=None, answer_dic=None):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    
    with open(os.path.join(root, f'v2_mscoco_{split}2014_annotations.json'), 'r') as f:
        answer_data = json.load(f)

    with open(os.path.join(root, f'v2_OpenEnded_mscoco_{split}2014_questions.json'), 'r') as f:
        questions_data = json.load(f)

    question_id_to_answer = {a['question_id']: a.get('multiple_choice_answer') for a in answer_data['annotations']}
    result = []
    word_index = 1

    for question in tqdm.tqdm(questions_data['questions']):
        words = nltk.word_tokenize(question['question'])
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question_id_to_answer.get(question.get('question_id'))

        try:
            answer = answer_dic[answer_word]
            result.append((question['image_id'], question_token, answer))

        except:
            print("Skipped: ", question['question_id'])


    with open('../data/{}.pkl'.format(split), 'wb') as f:
        pickle.dump(result, f)

    return word_dic

def process_targ_question(root, split, word_dic=None, answer_dic=None):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    
    df = pd.read_csv(os.path.join(root, f"{split}_questions.csv"))
    result = []
    word_index = 1

    for question in tqdm.tqdm(df):
        words = nltk.word_tokenize(question['question'])
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question.get('answer')
        answer = answer_dic[answer_word]

        result.append((question['image'], question_token, answer))

    with open('../data/targ_{}.pkl'.format(split), 'wb') as f:
        pickle.dump(result, f)

    return word_dic

if __name__ == '__main__':
    root = sys.argv[1]
    test_root = 'raw_data/test'

    nltk.download('punkt')

    ans_dic = load_answer_dict("common_vocab.txt")
    word_dic = process_question(root, 'train', answer_dic=ans_dic)
    process_question(root, 'val', word_dic, ans_dic)
    process_targ_question(test_root, 'train', word_dic, ans_dic)
    process_targ_question(test_root, 'test', word_dic, ans_dic)

    with open('../data/dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': ans_dic}, f)