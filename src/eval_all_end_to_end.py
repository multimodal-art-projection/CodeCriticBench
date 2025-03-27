# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
from string import Template

import numpy as np
from sklearn.metrics import mean_squared_error


class Eval_CodeCriticBench:
    '''
        original dataset (9k): codecriticbench9k
            - prompt key name for requesting models: `question`
            - basic evaluation label key name: `correctness`
            - advanced evaluation label key name: `rating`, `checklist_rating`
            - bug evaluation label key name: `error_lists[i][0]` (only test big category)
                - big_error = ['Configuration Management Error', 'Data Management Error', 'Input Validation and Data Processing Error',
                    'Monitoring and Logging Management Error', 'Environment Variable Error', 'Dependency Management Error', 'Syntax Error',
                    'Design Flaw', 'Security Vulnerability', 'Log Security Issue', 'Reference Error', 'Session Management Error',
                    'Code Quality and Maintenance Error', 'Logic Error', 'Testing and Verification Error',  'Network and Communication Error',
                    'Exception Handling Error', 'User Permission and Authentication Error', 'File and I/O Error', 'Type Error',
                    'Internationalization and Localization Error', 'Performance Issue', 'Concurrency and Multithreading Error']
        for more fine-grained subsets evaluation, set `check_subset=True` and use the following key names:
            - [`source` | `category`]:
                - use `source` key name for 'Code Gen': ['mbpp', 'codeforce', 'live-code-bench', 'debug']
                - use `category` key name for 'Code QA': ['Fundamental Programming', 'Advanced Programming',
                    'Software Engineering', 'Data Analysis', 'Mathematics', 'Desktop and Web Development',
                    'Machine Learning', 'Scientific Computing', 'Databases', 'Multimedia', 'Operating Systems']
            - `checklist_category`
                - 'Code Gen': ['Correctness Verification', 'Code Readability Enhancement', 'Robustness Validation',
                    'Comprehensive Testing', 'Space Complexity Control', 'Code Style Consistency', 'Output Format Compliance',
                    'Maintainability', 'Time Complexity Optimization', 'Algorithm Optimization']
                - 'Code QA': ['Depth', 'Logical Coherence', 'Innovation', 'Practicality', 'Clarity', 'Reliability',
                    'Completeness', 'Maintainability', 'Correctness', 'Performance']
    '''
    code_gen_subset = ['mbpp', 'codeforce', 'live-code-bench', 'debug']
    code_qa_subset = ['Fundamental Programming', 'Advanced Programming', 'Software Engineering', 'Data Analysis', 'Mathematics',
                      'Desktop and Web Development', 'Machine Learning', 'Scientific Computing', 'Databases', 'Multimedia', 'Operating Systems']
    error_subset = ['Configuration Management Error', 'Data Management Error', 'Input Validation and Data Processing Error',
                    'Monitoring and Logging Management Error', 'Environment Variable Error', 'Dependency Management Error',
                    'Syntax Error', 'Design Flaw', 'Security Vulnerability', 'Log Security Issue', 'Reference Error',
                    'Session Management Error', 'Code Quality and Maintenance Error', 'Logic Error', 'Testing and Verification Error',
                    'Network and Communication Error', 'Exception Handling Error', 'User Permission and Authentication Error',
                    'File and I/O Error', 'Type Error', 'Internationalization and Localization Error', 'Performance Issue', 'Concurrency and Multithreading Error']

    template_str = Template(
    r"""
Input File: ${input_file_name}
-------------------- Basic Evaluation ACC (%) --------------------
+-------+---------+---------+
|  All  | CodeGen | Code QA |
+-------+---------+---------+
| ${acc_all} |  ${acc_all_gen}  |  ${acc_all_qa}  |
+-------------------------------------------+
| MBPP  | CodeForce | LiveCodeBench | Debug |
+-------+-----------+---------------+-------+
| ${acc_cg0} |   ${acc_cg1}   |     ${acc_cg2}     | ${acc_cg3} |
+---------------------------------------------------------------------------------------+
|  FP   |  AP   |  SE   |  DA   |  MA   |  DW   |  ML   |  SC   |  DB   |  MM   |  OS   |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
| ${acc_cqa0} | ${acc_cqa1} | ${acc_cqa2} | ${acc_cqa3} | ${acc_cqa4} | ${acc_cqa5} | ${acc_cqa6} | ${acc_cqa7} | ${acc_cqa8} | ${acc_cqa9} | ${acc_cqa10} |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+

-------------------- Advanced Evaluation MSE --------------------
+-------+---------+---------+
|  All  | CodeGen | Code QA |
+-------+---------+---------+
| ${mse_all} |  ${mse_all_gen}  |  ${mse_all_qa}  |
+-------------------------------------------+
| MBPP  | CodeForce | LiveCodeBench | Debug |
+-------+-----------+---------------+-------+
| ${mse_cg0} |   ${mse_cg1}   |     ${mse_cg2}     | ${mse_cg3} |
+---------------------------------------------------------------------------------------+
|  FP   |  AP   |  SE   |  DA   |  MA   |  DW   |  ML   |  SC   |  DB   |  MM   |  OS   |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
| ${mse_cqa0} | ${mse_cqa1} | ${mse_cqa2} | ${mse_cqa3} | ${mse_cqa4} | ${mse_cqa5} | ${mse_cqa6} | ${mse_cqa7} | ${mse_cqa8} | ${mse_cqa9} | ${mse_cqa10} |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
For more fine-grained subset across different dimensions, split data according to subset name and dimension name

-------------------- Debug Evaluation Pass@1 ACC --------------------
+-------+-------+-------+-------+-------+-------+-------+-------+
|  All  |  CME  |  DME  | IVDPE | MLME  |  EVE  |  DME  |  SE   |
+-------+-------+-------+-------+-------+-------+-------+-------+
| ${acc_dbg_all} | ${acc_dbg0} | ${acc_dbg1} | ${acc_dbg2} | ${acc_dbg3} | ${acc_dbg4} | ${acc_dbg5} | ${acc_dbg6} |
+-------+-------+-------+-------+-------+-------+-------+-------+
|  DF   |  SV   |  LSI  |  RE   |  SME  |  CQME |  LE   |  TVE  |
+-------+-------+-------+-------+-------+-------+-------+-------+
| ${acc_dbg7} | ${acc_dbg8} | ${acc_dbg9} | ${acc_dbg10} | ${acc_dbg11} | ${acc_dbg12} | ${acc_dbg13} | ${acc_dbg14} |
+-------+-------+-------+-------+-------+-------+-------+-------+
|  NCE  |  EHE  |  UPAE |  FIE  |  TE   |  ILE  |  PI   |  CME  |
+-------+-------+-------+-------+-------+-------+-------+-------+
| ${acc_dbg15} | ${acc_dbg16} | ${acc_dbg17} | ${acc_dbg18} | ${acc_dbg19} | ${acc_dbg20} | ${acc_dbg21} | ${acc_dbg22} |
+-------+-------+-------+-------+-------+-------+-------+-------+
    """
    )
    def __init__(self, input_file_name=None, prompt_key_name=None, output_key_name=None):
        self.input_file_name = input_file_name
        self.prompt_key_name = prompt_key_name
        self.output_key_name = output_key_name

        self.basic_data, self.advanced_data, self.bug_data = self._split_data(input_file_name, prompt_key_name)
        self.acc_all, self.acc_gen_all, self.acc_qa_all = -1, -1, -1
        self.mse_all, self.mse_gen_all, self.mse_qa_all = -1, -1, -1
        self.acc_debug_all = -1
        self.acc_gen, self.acc_qa = {}, {}
        self.mse_gen, self.mse_qa = {}, {}
        self.acc_debug = {}

    def _split_data(self, input_file_name, prompt_key_name):
        assert input_file_name is not None and len(input_file_name) > 0
        basic_data, advanced_data, bug_data = [], [], []
        with open(input_file_name, 'r', encoding='utf-8') as fp:
            for line in fp:
                curr_data = json.loads(line.strip())
                if 'is_assistant_correct' in curr_data[prompt_key_name].lower():
                    basic_data.append(curr_data)
                elif 'comprehensive score' in curr_data[prompt_key_name].lower():
                    advanced_data.append(curr_data)
                else:
                    bug_data.append(curr_data)
        return basic_data, advanced_data, bug_data

    @staticmethod
    def _extract_correctness(text):
        start_pos = text.find('is_assistant_correct')
        end_pos = text.find('}', start_pos + len('is_assistant_correct'))
        predict_text = text[start_pos + len('is_assistant_correct'):end_pos]
        return 'Correct' if 'correct' in predict_text.lower() else 'Error'

    @staticmethod
    def _extract_checklist_ratings(text):
        pattern = r'Score:[\s\S]*?(\d+)'
        matches = re.findall(pattern, text)
        rating = []
        for match in matches:
            rating.append(int(match))
        return [0] * 11 if len(rating) == 0 else rating

    @staticmethod
    def _extract_final_rating(text):
        start_pos = text.find('comprehensive score')
        predict_text = text[start_pos + len('comprehensive score'):]
        match = re.search(r'\d+', predict_text)
        return int(match.group()) if match else 0

    @staticmethod
    def _cal_acc(grounds, predicts):
        same = 0
        for i in range(len(grounds)):
            if grounds[i] == predicts[i]:
                same += 1
        return same / len(grounds) if len(grounds) != 0 else -1

    @staticmethod
    def _cal_mse(grounds, predicts):
        '''
            similar metrics: mae, rmse, sd

            from sklearn.metrics import mean_squared_error, mean_absolute_error
            mae = mean_absolute_error(grounds, predicts)
            mse = mean_squared_error(grounds, predicts)
            rmse = np.sqrt(mse)
            errors = grounds - predicts
            sd = np.std(errors)
        '''
        grounds, predicts = np.array(grounds), np.array(predicts)
        mse = mean_squared_error(grounds, predicts)
        return mse

    def _eval_basic_coarse(self):
        assert self.basic_data is not None and len(self.basic_data) > 0

        subset = ['all', 'gen', 'qa']
        subset_gp = {}
        for name in subset:
            subset_gp[name] = {'grounds': [], 'predicts': []}
        for curr in self.basic_data:
            ground = 1 if 'correct' in curr['correctness'].lower() else 0
            predict = 1 if 'correct' in self._extract_correctness(curr[self.output_key_name]).lower() else 0
            if 'stackoverflow' not in curr['source']:
                subset_gp['gen']['grounds'].append(ground)
                subset_gp['gen']['predicts'].append(predict)
            else:
                subset_gp['qa']['grounds'].append(ground)
                subset_gp['qa']['predicts'].append(predict)
            subset_gp['all']['grounds'].append(ground)
            subset_gp['all']['predicts'].append(predict)

        self.acc_all = self._cal_acc(subset_gp['all']['grounds'], subset_gp['all']['predicts'])
        self.acc_gen_all = self._cal_acc(subset_gp['gen']['grounds'], subset_gp['gen']['predicts'])
        self.acc_qa_all = self._cal_acc(subset_gp['qa']['grounds'], subset_gp['qa']['predicts'])

    def _eval_basic_fine(self):
        assert self.basic_data is not None and len(self.basic_data) > 0

        code_gen_gp, code_qa_gp = {}, {}
        for name in self.code_gen_subset:
            code_gen_gp[name] = {'grounds': [], 'predicts': []}
        for name in self.code_qa_subset:
            code_qa_gp[name] = {'grounds': [], 'predicts': []}

        for curr in self.basic_data:
            if curr['source'] != 'stackoverflow':
                name = curr['source']
                code_gen_gp[name]['grounds'].append(1 if 'correct' in curr['correctness'].lower() else 0)
                code_gen_gp[name]['predicts'].append(1 if 'correct' in self._extract_correctness(curr[self.output_key_name]).lower() else 0)
            else:
                name = curr['category']
                code_qa_gp[name]['grounds'].append(1 if 'correct' in curr['correctness'].lower() else 0)
                code_qa_gp[name]['predicts'].append(1 if 'correct' in self._extract_correctness(curr[self.output_key_name]).lower() else 0)
        for name in self.code_gen_subset:
            self.acc_gen[name] = self._cal_acc(code_gen_gp[name]['grounds'], code_gen_gp[name]['predicts'])
        for name in self.code_qa_subset:
            self.acc_qa[name] = self._cal_acc(code_qa_gp[name]['grounds'], code_qa_gp[name]['predicts'])

    def _eval_advanced_coarse(self):
        assert self.advanced_data is not None and len(self.advanced_data) > 0

        subset = ['all', 'gen', 'qa']
        subset_gp = {}
        for name in subset:
            subset_gp[name] = {'grounds': [], 'predicts': []}
        for curr in self.advanced_data:
            ground = curr['rating']
            predict_rating = self._extract_checklist_ratings(curr[self.output_key_name])

            # illegal model output processing
            if 10 < predict_rating[-1] < 20:
                predict_rating = 10
            elif 20 < predict_rating[-1] <= 100:
                predict_rating = int(predict_rating[-1] / 10)
            elif 100 < predict_rating[-1]:
                scale = 10 ** (int(math.log10(predict_rating[-1])) - 1)
                predict_rating = int(predict_rating[-1] / scale)
            else:
                predict_rating = predict_rating[-1]

            if predict_rating == 0:
                predict_rating = self._extract_final_rating(curr[self.output_key_name])
                if 20 < predict_rating <= 100:
                    predict_rating = int(predict_rating / 10)
                elif 100 < predict_rating:
                    scale = 10 ** (int(math.log10(predict_rating)))
                    predict_rating = int(predict_rating / scale)

            if 'stackoverflow' not in curr['source']:
                subset_gp['gen']['grounds'].append(ground)
                subset_gp['gen']['predicts'].append(predict_rating)
            else:
                subset_gp['qa']['grounds'].append(ground)
                subset_gp['qa']['predicts'].append(predict_rating)
            subset_gp['all']['grounds'].append(ground)
            subset_gp['all']['predicts'].append(predict_rating)

        self.mse_all = self._cal_mse(subset_gp['all']['grounds'], subset_gp['all']['predicts'])
        self.mse_gen_all = self._cal_mse(subset_gp['gen']['grounds'], subset_gp['gen']['predicts'])
        self.mse_qa_all = self._cal_mse(subset_gp['qa']['grounds'], subset_gp['qa']['predicts'])

    def _eval_advanced_fine(self):
        assert self.advanced_data is not None and len(self.advanced_data) > 0

        code_gen_gp, code_qa_gp = {}, {}
        for name in self.code_gen_subset:
            code_gen_gp[name] = {'grounds': [], 'predicts': []}
        for name in self.code_qa_subset:
            code_qa_gp[name] = {'grounds': [], 'predicts': []}

        for curr in self.advanced_data:
            predict_rating = self._extract_checklist_ratings(curr[self.output_key_name])

            # illegal model output processing
            if 10 < predict_rating[-1] < 20:
                predict_rating = 10
            elif 20 < predict_rating[-1] <= 100:
                predict_rating = int(predict_rating[-1] / 10)
            elif 100 < predict_rating[-1]:
                scale = 10 ** (int(math.log10(predict_rating[-1])) - 1)
                predict_rating = int(predict_rating[-1] / scale)
            else:
                predict_rating = predict_rating[-1]

            if predict_rating == 0:
                predict_rating = self._extract_final_rating(curr[self.output_key_name])
                if 20 < predict_rating <= 100:
                    predict_rating = int(predict_rating / 10)
                elif 100 < predict_rating:
                    scale = 10 ** (int(math.log10(predict_rating)))
                    predict_rating = int(predict_rating / scale)

            if curr['source'] != 'stackoverflow':
                name = curr['source']
                code_gen_gp[name]['grounds'].append(curr['rating'])
                code_gen_gp[name]['predicts'].append(predict_rating)
            else:
                name = curr['category']
                code_qa_gp[name]['grounds'].append(curr['rating'])
                code_qa_gp[name]['predicts'].append(predict_rating)

        for name in self.code_gen_subset:
            self.mse_gen[name] = self._cal_mse(code_gen_gp[name]['grounds'], code_gen_gp[name]['predicts'])
        for name in self.code_qa_subset:
            self.mse_qa[name] = self._cal_mse(code_qa_gp[name]['grounds'], code_qa_gp[name]['predicts'])

    def _eval_bug_coarse(self):
        assert self.bug_data is not None and len(self.bug_data) > 0
        TP, TN, FP, FN = 0, 0, 0, 0
        for curr in self.bug_data:
            ground = [curr['error_lists'][i][0] for i in range(len(curr['error_lists']))]
            predict_text = curr[self.output_key_name]

            pred_start = predict_text.find('Category')
            pred_end = predict_text.find('\n', pred_start)
            predict_text = predict_text[pred_start + len('Category'):pred_end]

            # pass@1
            if any(e.lower() in predict_text.lower() for e in ground):
                TP += 1
            else:
                FN += 1
        acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else -1
        self.acc_debug_all = acc

    def _eval_bug_fine(self):
        assert self.bug_data is not None and len(self.bug_data) > 0

        subset_gp = {}
        for name in self.error_subset:
            subset_gp[name] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

        for curr in self.bug_data:
            ground = [curr['error_lists'][i][0] for i in range(len(curr['error_lists']))]
            predict_text = curr[self.output_key_name]

            pred_start = predict_text.find('Category')
            pred_end = predict_text.find('\n', pred_start)
            predict_text = predict_text[pred_start + len('Category'):pred_end]

            for e in ground:
                if e.lower() in predict_text.lower():
                    subset_gp[e]['TP'] += 1
                else:
                    subset_gp[e]['FN'] += 1

        for name in self.error_subset:
            numerator, denominator = subset_gp[name]['TP'] + subset_gp[name]['TN'], \
                subset_gp[name]['TP'] + subset_gp[name]['TN'] + subset_gp[name]['FP'] + subset_gp[name]['FN']
            self.acc_debug[name] = numerator / denominator if denominator != 0 else -1

    def report(self):
        self._eval_basic_coarse()
        self._eval_basic_fine()
        self._eval_advanced_coarse()
        self._eval_advanced_fine()
        self._eval_bug_coarse()
        self._eval_bug_fine()

        substitution_dict = {
            'input_file_name': self.input_file_name,
            'acc_all': f"{self.acc_all * 100:>5.2f}" if self.acc_all != -1 else 'N/A',
            'acc_all_gen': f"{self.acc_gen_all * 100:>5.2f}" if self.acc_gen_all != -1 else 'N/A',
            'acc_all_qa': f"{self.acc_qa_all * 100:>5.2f}" if self.acc_qa_all != -1 else 'N/A',
            'mse_all': f"{self.mse_all:>5.2f}" if self.mse_all != -1 else 'N/A',
            'mse_all_gen': f"{self.mse_gen_all:>5.2f}" if self.mse_gen_all != -1 else 'N/A',
            'mse_all_qa': f"{self.mse_qa_all:>5.2f}" if self.mse_qa_all != -1 else 'N/A',
            'acc_dbg_all': f"{self.acc_debug_all * 100:>5.2f}" if self.acc_debug_all != -1 else 'N/A',
        }

        for i, subset in enumerate(self.code_gen_subset):
            substitution_dict[f'acc_cg{i}'] = f"{self.acc_gen.get(subset, -1) * 100:>5.2f}" if self.acc_gen.get(subset, -1) != -1 else 'N/A'
            substitution_dict[f'mse_cg{i}'] = f"{self.mse_gen.get(subset, -1):>5.2f}" if self.mse_gen.get(subset, -1) != -1 else 'N/A'

        for i, subset in enumerate(self.code_qa_subset):
            substitution_dict[f'acc_cqa{i}'] = f"{self.acc_qa.get(subset, -1) * 100:>5.2f}" if self.acc_qa.get(subset, -1) != -1 else 'N/A'
            substitution_dict[f'mse_cqa{i}'] = f"{self.mse_qa.get(subset, -1):>5.2f}" if self.mse_qa.get(subset, -1) != -1 else 'N/A'

        for i, subset in enumerate(self.error_subset):
            substitution_dict[f'acc_dbg{i}'] = f"{self.acc_debug.get(subset, -1) * 100:>5.2f}" if self.acc_debug.get(subset, -1) != -1 else 'N/A'

        report_str = self.template_str.substitute(substitution_dict)
        print(report_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('eval the codecriticbench dataset (total/subsets) (basic/advanced/bug) (coarse-grained)')
    parser.add_argument('--input_file_name', type=str, default='./output_claude35.jsonl')
    parser.add_argument('--prompt_key_name', type=str, default='question')
    parser.add_argument('--output_key_name', type=str, default='generated')
    args = parser.parse_args()

    input_file_name = args.input_file_name
    prompt_key_name = args.prompt_key_name
    output_key_name = args.output_key_name

    #input_file_name = './output_claude35.jsonl'
    #prompt_key_name = 'question'
    #output_key_name = 'generated'

    eval_codecriticbench = Eval_CodeCriticBench(input_file_name, prompt_key_name, output_key_name)
    eval_codecriticbench.report()
