import os
import sys
import json
import pickle

import nltk
import tqdm
from PIL import Image

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
    answer_index = 0

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

        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        result.append((question['image_id'], question_token, answer))

    with open('../data/{}.pkl'.format(split), 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic

if __name__ == '__main__':
    root = sys.argv[1]

    nltk.download('punkt')
    word_dic, answer_dic = process_question(root, 'train')
    process_question(root, 'val', word_dic, answer_dic)

    with open('../data/dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)