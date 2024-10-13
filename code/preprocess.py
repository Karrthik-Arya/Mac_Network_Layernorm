import os
import sys
import json
import pickle
import re

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

def preprocessing(text):
  input_text = text
  input_text = input_text.lower()

  # Removing periods except if it occurs as decimal
  input_text = re.sub(r'(?<!\d)\.(?!\d)', '', input_text)

  # Converting number words to digits
  number_words = {
      "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
      "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
      "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
      "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
      "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
      "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
      "eighty": "80", "ninety": "90"
  }
  input_text = re.sub(r'\b(?:' + '|'.join(number_words.keys()) + r')\b', lambda x: number_words[x.group()], input_text)

  # Removing articles (a, an, the)
  if len(input_text)>3:
    input_text = re.sub(r'\b(?:a|an|the)\b', '', input_text)

  # Adding apostrophe if a contraction is missing it
  input_text = re.sub(r'\b(\w+(?<!e)(?<!a))nt\b', r"\1n't", input_text)

  # input_text = re.sub(r'\b(\w+(?<!t))ent\b', r"\1en't", input_text)

  # Replacing all punctuation (except apostrophe and colon) withinput_text a space character
  input_text = re.sub(r'[^\w\':]|(?<=\d),(?=\d)', ' ', input_text)

  # Removing extra spaces
  input_text = re.sub(r'\s+', ' ', input_text).strip()

  return input_text

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

        answer_word = preprocessing(question_id_to_answer.get(question.get('question_id')))

        try:
            answer = answer_dic[answer_word]
            result.append((question['image_id'], question_token, answer))

        except:
            pass
            # print("Skipped: ", question['question_id'])


    with open('data/{}.pkl'.format(split), 'wb') as f:
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

    for _, question in tqdm.tqdm(df.iterrows(), total=len(df)):
        words = nltk.word_tokenize(question['question'])
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = preprocessing(question.get('answer'))
        answer = answer_dic[answer_word]

        result.append((question['image'], question_token, answer))

    with open('data/targ_{}.pkl'.format(split), 'wb') as f:
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

    with open('data/dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': ans_dic}, f)