'''
This code file is used to write model outputs and human ratings into JSON format for the purpose of making the data publicly available.
'''

from Extract_Scores import list_excel_files, get_excel_sheet_names, get_original_scores, sort_scores, check, human_allocation_A, human_allocation_B, steps_dimensions_wo_integrity
import os
from docx import Document
import json
from openpyxl.utils import coordinate_to_tuple
responses_lst_A = [f"A{i:02d}_FS{j}" for i in range(1, 11) for j in range(1, 11)]
# print(responses_lst_A)
responses_lst_B = [f"B{i:02d}_FS{j}" for i in range(1, 11) for j in range(1, 11)]
import pandas as pd
import numpy as np

def list_challenge_and_solution_with_num(text):
    '''
    List the text content of Step-1 or Step-3, organized by scenario/challenge ID.
    '''
    lines, content = text.split('\n'), []
    start = 1
    for line in lines:
        if line.startswith(str(start)):
            content.append(line.lstrip())
            start += 1

    return content


def get_response(Response_ID):
    """
    Input: Answer ID, e.g., "A01_FS1"
    Output: in the following format:
    [
        ['A01_FS1'],
        ['Step-1 Responses'],
        ...
        ['Step-6 Responses']
    ]
    """

    doc = Document(os.path.join('Answers', 'Answer', Response_ID[:3], Response_ID + '.docx'))
    for table in doc.tables:
        table_content = []
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells]
            table_content.append(row_text)
        # print(table_content)
    return table_content


locations = {
    'Step-1': [
        {'Category': 'C5', 'Invalid': 'H5', 'Elaboration': 'P5', 'Originality': 'Q5'}, 
        {'Category': 'C6', 'Invalid': 'H6', 'Elaboration': 'P6', 'Originality': 'Q6'}, 
        {'Category': 'C7', 'Invalid': 'H7', 'Elaboration': 'P7', 'Originality': 'Q7'}, 
        {'Category': 'C8', 'Invalid': 'H8', 'Elaboration': 'P8', 'Originality': 'Q8'}, 
        {'Category': 'C9', 'Invalid': 'H9', 'Elaboration': 'P9', 'Originality': 'Q9'}, 
        {'Category': 'C10', 'Invalid': 'H10', 'Elaboration': 'P10', 'Originality': 'Q10'}, 
        {'Category': 'C11', 'Invalid': 'H11', 'Elaboration': 'P11', 'Originality': 'Q11'}, 
        {'Category': 'C12', 'Invalid': 'H12', 'Elaboration': 'P12', 'Originality': 'Q12'}
    ],
    'Step-3': [
        {'Category': 'C56', 'Invalid': 'H56', 'Elaboration': 'P56', 'Originality': 'Q56'},
        {'Category': 'C57', 'Invalid': 'H57', 'Elaboration': 'P57', 'Originality': 'Q57'},
        {'Category': 'C58', 'Invalid': 'H58', 'Elaboration': 'P58', 'Originality': 'Q58'},
        {'Category': 'C59', 'Invalid': 'H59', 'Elaboration': 'P59', 'Originality': 'Q59'},
        {'Category': 'C60', 'Invalid': 'H60', 'Elaboration': 'P60', 'Originality': 'Q60'},
        {'Category': 'C61', 'Invalid': 'H61', 'Elaboration': 'P61', 'Originality': 'Q61'},
        {'Category': 'C62', 'Invalid': 'H62', 'Elaboration': 'P62', 'Originality': 'Q62'},
        {'Category': 'C63', 'Invalid': 'H63', 'Elaboration': 'P63', 'Originality': 'Q63'}
    ],
    'Step-4': [
        {'Correctly Written': 'C88', 'Relevance': 'F88'},
        {'Correctly Written': 'C89', 'Relevance': 'F89'},
        {'Correctly Written': 'C90', 'Relevance': 'F90'},
        {'Correctly Written': 'C91', 'Relevance': 'F91'},
        {'Correctly Written': 'C92', 'Relevance': 'F92'}
    ]
}


def get_complete_response(file_path):
    '''
    This function traverses all Excel rating sheet files and worksheets
    within the 'Human Evaluation' folder, and extracts detailed scores
    for all responses in Step-1, Step-3, and Step-4.
    For example, the categories corresponding to each challenge.

    Inputs: Relative path to the folder, e.g., 'Human Evaluation'
    Outputs: In the following format:
        [
            [
                Answer_ID, Human_ID, FS_ID,
                {
                    'Step-1': [
                        {'Category': 'category content', 'Originality': score, 'Elaboration': score / 'Invalid': 'reason if invalid'},
                        ...
                    ],
                    'Step-3': [
                        {'Category': 'category content', 'Originality': score, 'Elaboration': score / 'Invalid': 'reason if invalid'},
                        ...
                    ],
                    'Step-4': [
                        {'Correctly Written': score, 'Relevance': score},
                        ...
                    ]
                }
            ],
            ...
        ]
    '''
    CScore = []
    
    for f in list_excel_files(file_path):  # each rating sheet file
        file = os.path.join(file_path, f)
        sheets = get_excel_sheet_names(file)
        assert all('_' in item for item in sheets), file + ' has an issue with its worksheets!'
        
        for sheet in sheets:
            df = pd.read_excel(file, sheet_name=sheet, header=None)

            def get_value(cell):
                row_idx, col_idx = coordinate_to_tuple(cell)
                return df.iat[row_idx - 1, col_idx - 1]

            TScore = [
                sheet,
                f[:-5].split("_")[0],
                f[:-5].split("_")[1]
            ]
            step_dic = {}

            for step in ['Step-1', 'Step-3']:
                Slst = []
                for dimension in locations[step]:
                    Ddic = {}
                    assert pd.isna(get_value(dimension['Category'])) ^ pd.isna(get_value(dimension['Invalid'])), f + '  ' + sheet + '  ' + step + " Category and Invalid must have exactly one empty (NaN)"
                    if pd.isna(get_value(dimension['Category'])):
                        Ddic['Invalid'] = translate(get_value(dimension['Invalid']), step)
                    elif pd.isna(get_value(dimension['Invalid'])):
                        Ddic['Category'] = translate(get_value(dimension['Category']), step)
                        assert not pd.isna(get_value(dimension['Originality'])), f + '  ' + sheet + '  ' + step + " Originality cannot be NaN"
                        assert not pd.isna(get_value(dimension['Elaboration'])), f + '  ' + sheet + '  ' + step + " Elaboration cannot be NaN"
                        Ddic['Originality'] = round(get_value(dimension['Originality']))
                        Ddic['Elaboration'] = round(get_value(dimension['Elaboration']))
                    Slst.append(Ddic)
                step_dic[step] = Slst
            
            # Special handling for Step-4
            Slst = []
            for dimension in locations['Step-4']:
                Ddic = {}
                assert (
                    not pd.isna(get_value(dimension['Correctly Written'])) 
                    and 
                    not pd.isna(get_value(dimension['Relevance']))
                ), "Correctly Written and Relevance cannot be NaN"
                Ddic['Correctly Written'] = round(get_value(dimension['Correctly Written']))
                Ddic['Relevance'] = round(get_value(dimension['Relevance']))
                Slst.append(Ddic)
            step_dic['Step-4'] = Slst

            TScore.append(step_dic)
            CScore.append(TScore)

    assert len(CScore) == 200, 'The number of extracted detailed scores is not 200!'
    return CScore



def translate(text, step):
    if '（' in text and '）' in text:
        return text[text.find("（") + 1 : text.find("）")]
    elif '(' in text and ')' in text:
        return text[text.find("(") + 1 : text.find(")")]
    else:
        if step == 'Step-1':
            invalid = {
                "P — 含义模糊不清": "Perhaps — Ambiguous",
                "W — 挑战与未来情境无关": "Why — Irrelevant to Scenario",
                "S — 陈述为某挑战的解决方案": "Solution — A Solution to a Challenge",
                "D — 挑战与另一条YES挑战过于相似": "Duplicate — Similar to Another YES Challenge",
                "B — 未提供作答内容": "Blank — No Response Provided"
            }
        else:
            invalid = {
                "P — 方案与KVP和目的的关系不清晰": "Perhaps — The relationship between the solution, the KVP, and the purpose is unclear",
                "W — 方案与潜在问题无关": "Why — The solution is irrelevant to the underlying problem",
                "D — 该方案与另一条Yes方案过于相似": "Duplicate — The solution is too similar to another YES solution",
                "B — 未提供作答内容": "Blank — No Response Provided"
            }
        return invalid[text]


def write_json(condition):
    '''
    Input: 'A' for TeaMAC or 'B' for Baseline
    '''
    all_responses_id = responses_lst_A if condition == 'A' else responses_lst_B
    dir_name = 'DATA_ACC' if condition == 'A' else 'DATA_BCC'
    human_allocation = human_allocation_A if condition == 'A' else human_allocation_B
    results = []

    raw_scores_C = get_original_scores(os.path.join('Human Evaluation', dir_name))
    raw_complete = get_complete_response(os.path.join('Human Evaluation', dir_name))
    
    check(raw_scores_C, human_allocation)

    scores_after_sort_C = sort_scores(raw_scores_C)
    complete_after_sort_C = sort_scores(raw_complete)

    all_json = []

    for Response_ID in all_responses_id:
        response = get_response(Response_ID)  # Get the content for this response
        scores = [item for item in scores_after_sort_C if item[0] == Response_ID]
        completes = [item for item in complete_after_sort_C if item[0] == Response_ID]
        assert len(scores) == 2, Response_ID + ' does not have 2 scores!'
        assert len(completes) == 2, Response_ID + ' does not have 2 complete entries!'
        assert len(set([completes[0][0], completes[1][0], scores[0][0], scores[1][0]])) == 1, 'scores and complete do not match!'

        each_json = [
            Response_ID,
            [scores[0][1], scores[1][1]],
            scores[0][2]
        ]
        
        for human_id in range(len(each_json[1])):
            each_human_json = {}
            for step in steps_dimensions_wo_integrity:
                each_step = []
                if step in ['Step-1', 'Step-3', 'Step-4']:
                    all_responses = list_challenge_and_solution_with_num(response[int(step[-1])][0])
                    num = 8 if step in ['Step-1', 'Step-3'] else 5
                    if len(all_responses) > num:
                        print(f'Warning: {Response_ID} has more responses than expected for {step}!')
                        for i in range(num):
                            each_response = {'Content': all_responses[i]}
                            assert len(completes[human_id][3][step]) == num, 'Number of complete responses is incorrect!'
                            each_response |= completes[human_id][3][step][i]
                            each_step.append(each_response)
                    else:
                        for i in range(len(all_responses)):
                            each_response = {'Content': all_responses[i]}
                            assert len(completes[human_id][3][step]) == num, 'Number of complete responses is incorrect!'
                            each_response |= completes[human_id][3][step][i]
                            each_step.append(each_response)
                        if i < num - 1:
                            for j in range(i + 1, num):
                                each_response = {'Content': ''}
                                assert len(completes[human_id][3][step]) == num, 'Number of complete responses is incorrect!'
                                each_response |= completes[human_id][3][step][i]
                                each_step.append(each_response)

                    each_step.append({'Score Summary': scores[human_id][3][step]})
                else:
                    each_response = {'Content': response[int(step[-1])][0]}
                    each_response |= scores[human_id][3][step]
                    each_step.append(each_response)
                each_human_json[step] = each_step
            each_human_json['Total Score'] = scores[human_id][3]['Total Score']
            each_json.append(each_human_json)

        all_json.append(each_json)

    with open(os.path.join('Datasets', f'data_{condition}.json'), 'w', encoding='utf-8') as f:
        json.dump(all_json, f, ensure_ascii=False, indent=4)


for type in ['A', 'B']:
    write_json(type)
