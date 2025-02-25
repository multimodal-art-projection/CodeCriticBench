# -*- coding: utf-8 -*-

import json
import math
import re
import numpy as np
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error


def extract_correctness(text):
    start_pos = text.find('is_assistant_correct')
    end_pos = text.find('}', start_pos + len('is_assistant_correct'))
    predict_text = text[start_pos + len('is_assistant_correct'):end_pos]
    return 'Correct' if 'correct' in predict_text.lower() else 'Error'

def extract_checklist_rating(text):
    pattern = r'Score:[\s\S]*?(\d+)'
    matches = re.findall(pattern, text)
    rating = []
    for match in matches: rating.append(int(match))
    return [0] * 11 if len(rating) == 0 else rating

def extract_final_rating(text):
    start_pos = text.find('comprehensive score')
    predict_text = text[start_pos + len('comprehensive score'):]
    match = re.search(r'\d+', predict_text)
    return int(match.group()) if match else 0

def extract_bug_rating(text):
    start_pos, end_pos = text.find('Rating'), text.find('Reason')
    predict_text = text[start_pos + len('Rating'):end_pos]
    match = re.search(r'\d+', predict_text)
    return int(match.group()) if match else 0

def cal_accuracy(ground, predict):
    same = 0
    for i in range(len(ground)):
        if ground[i] == predict[i]: same += 1
    return same / len(ground) if len(ground) != 0 else -1


def cal_mse(ground, predict):
    ground, predict = np.array(ground), np.array(predict)
    mse = mean_squared_error(ground, predict)
    return mse

def eval_level1(lines):
    all_ground, all_predict = [], []
    for line in lines:
        curr = json.loads(line.strip())

        ground_correctness = curr['correctness']
        predict_correctness = extract_correctness(curr['generated'])

        if 'correct' in ground_correctness.lower(): all_ground.append(1)
        else: all_ground.append(0)
        if 'correct' in predict_correctness.lower(): all_predict.append(1)
        else: all_predict.append(0)

    all_accuracy = cal_accuracy(all_ground, all_predict)
    print(f'{all_accuracy*100:.2f}')

def eval_level2(lines):
    regress = {'ground': [], 'predict': []}
    for line in lines:
        curr = json.loads(line.strip())

        ground_rating = curr['rating']
        predict_rating = extract_checklist_rating(curr['generated'])


        if 10 < predict_rating[-1] < 20: predict_rating = 10
        elif 20 < predict_rating[-1] <= 100: predict_rating = int(predict_rating[-1] / 10)
        elif 100 < predict_rating[-1]:
            scale = 10 ** (int(math.log10(predict_rating[-1])) - 1)
            predict_rating = int(predict_rating[-1] / scale)
        else: predict_rating = predict_rating[-1]

        if predict_rating == 0:
            predict_rating = extract_final_rating(curr['generated'])
            if 20 < predict_rating <= 100: predict_rating = int(predict_rating / 10)
            elif 100 < predict_rating:
                scale = 10 ** (int(math.log10(predict_rating)))
                predict_rating = int(predict_rating / scale)

        regress['ground'].append(ground_rating)
        regress['predict'].append(predict_rating)

    all_mse = cal_mse(regress['ground'], regress['predict'])
    print(f'{all_mse:.2f}')

ERROR_NAME = ['Configuration Management Error', 'Data Management Error', 'Input Validation and Data Processing Error', 'Monitoring and Logging Management Error', 'Environment Variable Error', 'Dependency Management Error', 'Syntax Error', 'Design Flaw', 'Security Vulnerability', 'Log Security Issue', 'Reference Error', 'Session Management Error', 'Code Quality and Maintenance Error', 'Logic Error', 'Testing and Verification Error', 'Network and Communication Error', 'Exception Handling Error', 'User Permission and Authentication Error', 'File and I/O Error', 'Type Error', 'Internationalization and Localization Error', 'Performance Issue', 'Concurrency and Multithreading Error']
ERROR_INDEX = {name: i for i, name in enumerate(ERROR_NAME)}

def eval_level3(lines):
    all_ground, all_predict = [], []
    TP, FP, TN, FN = 0, 0, 0, 0
    for line in lines:
        curr = json.loads(line.strip())
        ground = [curr['error_lists'][i][0] for i in range(len(curr['error_lists']))]
        pred_str = curr['generated']
        pred_start = pred_str.find('Category')
        pred_end = pred_str.find('\n', pred_start)
        pred_str = pred_str[pred_start+len('Category'):pred_end]

        # pass@1
        if any(e.lower() in pred_str.lower() for e in ground): TP += 1
        else: FN += 1
    ACC = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
    print(f"{ACC * 100:.2f}")

if __name__ == '__main__':
    # filename1 = './output/claude35_sonnet/critic_level1.jsonl'
    # filename2 = './output/claude35_sonnet/critic_level2.jsonl'
    # filename3 = './output/claude35_sonnet/critic_level3.jsonl'

    filename1 = './output/gpt4o/critic_level1.jsonl'
    filename2 = './output/gpt4o/critic_level2.jsonl'
    filename3 = './output/gpt4o/critic_level3.jsonl'
 
    with open(filename1, 'r', encoding='utf-8') as fp:
        lines1 = fp.readlines()
    with open(filename2, 'r', encoding='utf-8') as fp:
        lines2 = fp.readlines()
    with open(filename3, 'r', encoding='utf-8') as fp:
        lines3 = fp.readlines()

    algo_level1 = lines1[:3200]
    real_level1 = lines1[3200:]
    eval_level1(lines1)
    eval_level1(algo_level1)
    eval_level1(real_level1)
    
    algo_level2 = lines2[:3200]
    real_level2 = lines2[3200:]
    eval_level2(lines2)
    eval_level2(algo_level2)
    eval_level2(real_level2)

    eval_level3(lines3)