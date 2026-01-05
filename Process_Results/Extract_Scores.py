"""
This script extracts human scores from the rating sheets and computes the
average for double-blind ratings to obtain the final response scores.

The output format is as follows:

PScore = [
    [Answer ID, Human ID(s), FS ID,
        {
            'Step-1': {dimension1: score, dimension2: score, ...},
            'Step-2': {...},
            ...
            'Step-6': {...},
            'Total Score': total_score_value
        }
    ],
    ...
]
"""


''' All dimensions contained in each step (without merging the Step-2 "Integrity" dimensions) '''
steps_dimensions_wo_integrity = {
    "Step-1": [
        "Fluency", "Flexibility", "Elaboration", "Originality", 'Overall'
    ],
    "Step-2": [
        'Condition Phrase', 'Stem & KVP', 'Purpose', 
        'FS Parameters', 'Focus', 'Adequacy', 'Overall'
    ],
    'Step-3': [
        "Fluency", "Flexibility", "Elaboration", "Originality", 'Overall'
    ],
    'Step-4': [
        'Correctly Written', 'Relevance', 'Overall'
    ],
    'Step-5': [
        'Correctly Used', 'Overall'
    ],
    'Step-6': [
        'Relevance', 'Effectiveness', 'Criteria', 'Impact', 'Humaness', 'Development', 'Overall'
    ]
}

''' All dimensions contained in each step (with merged Step-2 "Integrity" dimensions) '''
steps_dimensions_with_integrity = {
    "Step-1": [
        "Fluency", "Flexibility", "Elaboration", "Originality", 'Overall'
    ],
    "Step-2": [
        'Integrity', 'Focus', 'Adequacy', 'Overall'
    ],
    'Step-3': [
        "Fluency", "Flexibility", "Elaboration", "Originality", 'Overall'
    ],
    'Step-4': [
        'Correctly Written', 'Relevance', 'Overall'
    ],
    'Step-5': [
        'Correctly Used', 'Overall'
    ],
    'Step-6': [
        'Relevance', 'Effectiveness', 'Criteria', 'Impact', 'Humaness', 'Development', 'Overall'
    ]
}

''' All dimensions contained in each step (without merging the Step-2 "Integrity" dimensions) '''
steps_dimensions_max_wo_integrity = {
    "Step-1": {
        "Fluency": 8, "Flexibility": 8, "Elaboration": 16, "Originality": 16, 'Overall': 48
    },
    "Step-2": {
        'Condition Phrase': 2, 'Stem & KVP': 3, 'Purpose': 2, 
        'FS Parameters': 2, 'Focus': 10, 'Adequacy': 10, 'Overall': 30
    },
    'Step-3': {
        "Fluency": 8, "Flexibility": 8, "Elaboration": 16, "Originality": 16, 'Overall': 48
    },
    'Step-4': {
        'Correctly Written': 5, 'Relevance': 15, 'Overall': 20
    },
    'Step-5': {
        'Correctly Used': 5, 'Overall': 5
    },
    'Step-6': {
        'Relevance': 5, 'Effectiveness': 5, 'Criteria': 5, 'Impact': 5, 'Humaness': 5, 'Development': 10, 'Overall': 35
    },
    'Total Score': 216
}

''' All dimensions contained in each step (with merged Step-2 "Integrity" dimensions) '''
steps_dimensions_max_with_integrity = {
    "Step-1": {
        "Fluency": 8, "Flexibility": 8, "Elaboration": 16, "Originality": 16, 'Overall': 48
    },
    "Step-2": {
        'Integrity': 10, 'Focus': 10, 'Adequacy': 10, 'Overall': 30
    },
    'Step-3': {
        "Fluency": 8, "Flexibility": 8, "Elaboration": 16, "Originality": 16, 'Overall': 48
    },
    'Step-4': {
        'Correctly Written': 5, 'Relevance': 15, 'Overall': 20
    },
    'Step-5': {
        'Correctly Used': 5, 'Overall': 5
    },
    'Step-6': {
        'Relevance': 5, 'Effectiveness': 5, 'Criteria': 5, 'Impact': 5, 'Humaness': 5, 'Development': 10, 'Overall': 35
    },
    'Total Score': 216
}


# model_name = {
#     '01': 'qwen3-235b-a22b-instruct-2507',
#     '02': 'qwen3-235b-a22b-thinking-2507',
#     '03': 'DeepSeek-V3-0324',
#     '04': 'Deepseek-r1',
#     '05': 'kimi-k2-0711-preview',
#     '06': 'llama-4-scout-17b-16e-instruct',
#     '07': 'gpt-4o',
#     '08': 'gpt-5',
#     '09': 'claude-opus-4-1-20250805',
#     '10': 'gemini-2.5-pro'
# }

model_name = {
    '01': 'qwen3-235b-instruct',
    '02': 'qwen3-235b-thinking',
    '03': 'deepseek-v3',
    '04': 'deepseek-r1',
    '05': 'kimi-k2',
    '06': 'llama-4-scout',
    '07': 'gpt-4o',
    '08': 'gpt-5',
    '09': 'claude-4-opus',
    '10': 'gemini-2.5-pro'
}

import os
import pandas as pd
from openpyxl.utils import coordinate_to_tuple
from scipy.stats import pearsonr
import numpy as np
from collections import defaultdict
from itertools import combinations
import pingouin as pg


location = {
    "Step-1": {
        "Fluency": 'W19', "Flexibility": 'W21', "Elaboration": 'W23', "Originality": 'W25', 'Overall': 'W27'
    },
    "Step-2": {
        'Condition Phrase': 'U32', 'Stem & KVP': 'U34', 'Purpose': 'U37', 
        'FS Parameters': 'U40', 'Focus': 'U43', 'Adequacy': 'U46', 'Overall': 'W49'
    },
    'Step-3': {
        "Fluency": 'W70', "Flexibility": 'W72', "Elaboration": 'W74', "Originality": 'W76', 'Overall': 'W78'
    },
    'Step-4': {
        'Correctly Written': 'X86', 'Relevance': 'X89', 'Overall': 'W95'
    },
    'Step-5': {
        'Correctly Used': 'X92', 'Overall': 'W97'
    },
    'Step-6': {
        'Relevance': 'Y102', 'Effectiveness': 'Y106', 'Criteria': 'Y110', 
        'Impact': 'Y114', 'Humaness': 'Y118', 'Development': 'Y122', 'Overall': 'W129'
    },
    'Total Score': 'X133',
    'Human ID': 'F2',
    'Answer ID': 'X2',
    'FS ID': 'R2'
}


def list_excel_files(folder_path: str) -> list: 
    '''
    # Return a list of all Excel file names (without paths) in the specified folder
    '''
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]
    return excel_files


def get_excel_sheet_names(file_path: str) -> list: 
    '''
    # Return a list of all sheet names in the specified Excel file
    '''
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    return sheet_names

# def read_cell_value(file_path: str, sheet_name: str, cell: str):
#     df = pd.read_excel(file_path, sheet_name = sheet_name, header = None)
#     row, col = coordinate_to_tuple(cell)

#     value = df.iat[row - 1, col - 1]
#     return value


ScoreName_lst = [f'Step-{i}' for i in range(1, 7)]
fs_order = [f"FS{i}" for i in range(1, 11)]

def get_original_scores(file_path: str) -> list:
    """
    This function traverses all Excel rating sheet files and all worksheets
    within the 'Human Evaluation' folder, and extracts the scores for each
    evaluation dimension at every step.

    Inputs:
        - Relative path to the folder, e.g., 'Human Evaluation'

    Outputs:
        - In the format of PScore
    """

    PScore = []
    
    for f in list_excel_files(file_path):
        file = os.path.join(file_path, f)
        sheets = get_excel_sheet_names(file)
        assert all('_' in item for item in sheets), file + 'There is an issue with the worksheet!'
        
        for sheet in sheets:
            df = pd.read_excel(file, sheet_name = sheet, header = None)

            def get_value(cell):
                row_idx, col_idx = coordinate_to_tuple(cell)
                return df.iat[row_idx - 1, col_idx - 1]

            # TScore = [
            #     get_value(location['Answer ID']),
            #     get_value(location['Human ID']),
            #     get_value(location['FS ID'])
            # ]
            TScore = [
                sheet,
                f[:-5].split("_")[0],
                f[:-5].split("_")[1]
            ]
            step_dic = {}

            for step in ScoreName_lst:
                Sdic = {}
                for dimension in location[step]:
                    # print(get_value(location[step][dimension]))
                    # print(f, sheet, step, dimension)
                    if np.isnan(get_value(location[step][dimension])):
                        print(f, sheet, step, dimension)
                    Sdic[dimension] = round(get_value(location[step][dimension]))
                step_dic[step] = Sdic

            step_dic['Total Score'] = round(get_value(location['Total Score'])) # Total Score
            TScore.append(step_dic)
            PScore.append(TScore)

    return PScore


def sort_scores(scores_wo_sort: list) -> list:
    '''
    Sort a list in PScore format by Answer ID.
    '''
    def sort_key(item):
        part1, part2 = item[0].split('_') if isinstance(item, list) else item.split('_')
        num1 = int(part1[1:])
        fs_num = int(part2[2:])
        # human_id = int(item[1][1:])
        return (fs_num, num1) # (fs_num, human_id, num1)
    return sorted(scores_wo_sort, key = sort_key)


def check(mock_scores: list, human_allocation: dict):
    '''
    Check the output of the get_original_scores() function, including:
    1. Whether the total score for each step is calculated correctly
    2. Whether the pre-assigned correspondence between human raters and future scenarios is correct
    3. Whether every response has been scored by two raters
    4. Whether all response IDs and future scenario IDs are consistent
    '''
    # Check 1
    p1 = 0
    for i, record in enumerate(mock_scores):
        su = 0  # Calculate the Total Score for this response
        answer, human, fs, scores = record
        for step in scores:  # Check whether the sum for each step is correct
            if step != 'Total Score':
                su += scores[step]['Overall']
                if sum(v for k, v in scores[step].items() if k != "Overall") != scores[step]['Overall']:
                    print('Record', i + 1, '<', step, '> calculation error. Answer ID:', answer, ', Human ID:', human, ', FS ID:', fs)
                    print('Correct total should be:', sum(v for k, v in scores[step].items() if k != "Overall"), 'instead of', scores[step]['Overall'], '\n')
                    p1 = 1
        if su != scores['Total Score']:
            p1 = 1
            print('Record', i + 1, '<Total Score> calculation error. Answer ID:', answer, ', Human ID:', human, ', FS ID:', fs)
            print('Correct total should be:', su, 'instead of', scores['Total Score'], '\n')
    if p1 == 0:
        print("All scores are calculated correctly!")

    # Check 2
    p2 = 0
    for humanid in human_allocation.keys():
        fs_lst = [record[2] for record in mock_scores if record[1] == humanid]
        if sorted(set(fs_lst)) != sorted(human_allocation[humanid]):
            print('Human ID:', humanid, 'has scored scenarios:', sorted(set(fs_lst)), ', which does not match the pre-assignment.\n')
            p2 = 1
    if p2 == 0:
        print("All human raters match the pre-assigned future scenarios!")

    # Check 3
    p3, AnswerID_lst = 0, set([r[0] for r in mock_scores])
    for an in AnswerID_lst:
        if len([tn for tn in mock_scores if tn[0] == an]) != 2:
            print("Answer ID", an, 'does not have exactly two raters, number of raters:', len([tn for tn in mock_scores if tn[0] == an]))
            p3 = 1
    if p3 == 0:
        print("All responses have been scored by two raters!")

    # Check 4
    p4 = 0
    for i, bn in enumerate(mock_scores):
        if bn[0].split('_')[1] != bn[2]:
            p4 = 1
            print('Record', i + 1, 'Answer ID does not match FS ID. Answer ID:', bn[0], ', FS ID:', bn[2], '\n')
    if p4 == 0:
        print("All Answer IDs match their corresponding FS IDs!")



def pairwise_average(Original_list: list) -> list:
    '''
    Compute the average score for each response from the double-blind ratings to obtain the final score.

    Inputs: Output from get_original_scores()

    Outputs: Same format as the output of get_original_scores(), except that the rater ID field is now a list containing the two raters who scored the response.
        '''
    AnswerID_lst = set([r[0] for r in Original_list])
    PScore = []

    def average(a1: int, a2: int) -> int | float:
        s = a1 + a2
        if s % 2 == 0:
            return s // 2
        else:
            return s / 2

    for an in AnswerID_lst:
        pair_scores = [tn for tn in Original_list if tn[0] == an]
        ans1, ans2 = pair_scores[0], pair_scores[1]

        assert len(pair_scores) == 2, an + '_ Fewer than two raters provided scores!'
        assert ans1[2] == ans2[2], an + '_ The two ratings correspond to different scenario IDs!'
        
        TScore = [an, [ans1[1], ans2[1]], ans1[2]]
        step_dic = {}
        for step in ScoreName_lst:
            Sdic = {}
            for dimension in location[step]:
                Sdic[dimension] = average(ans1[3][step][dimension], ans2[3][step][dimension])
            step_dic[step] = Sdic
        
        step_dic['Total Score'] = average(ans1[3]['Total Score'], ans2[3]['Total Score']) # Total Score
        TScore.append(step_dic)
        PScore.append(TScore)
    
    assert len(PScore) * 2 == len(Original_list)
    return PScore


def corr(PScore, H1, H1_FS, H2, H2_FS, display = 0):
    '''
    Rating consistency between H1 and H2 for the ten responses in the scenario H1_FS (= H2_FS):
    Pearson consistency: Normalize all dimension scores for each response, concatenate them into a single list per response, and compute the Pearson correlation.

    ICC consistency: Normalize all dimension scores for all responses, concatenate them into a matrix/list, and compute the Intraclass Correlation Coefficient (ICC).
    '''
    data = []
    assert H1_FS == H2_FS, 'The two raters evaluated different scenarios.'
    ranker1_ini, ranker2_ini = [an for an in PScore if an[1] == H1 and an[2] == H1_FS], [an for an in PScore if an[1] == H2 and an[2] == H2_FS]
    assert len(ranker1_ini) == 10, H1 + " has " + str(len(ranker1_ini)) + " responses for " + H1_FS + "."
    assert len(ranker2_ini) == 10, H2 + " has " + str(len(ranker2_ini)) + " responses for " + H2_FS + "."
    ranker1_answer_lst = [an[0] for an in ranker1_ini]
    ranker2_answer_lst = [an[0] for an in ranker2_ini]
    assert sorted(ranker1_answer_lst) == sorted(ranker2_answer_lst), "The two raters have different response IDs for " + H1_FS + "."
    P, R = [], []
    for ansid in ranker1_answer_lst:
        R1, R2 = [], []
        for step in steps_dimensions_wo_integrity.keys():
            for dimension in [an for an in steps_dimensions_wo_integrity[step] if an != 'Overall']:
                dmax = steps_dimensions_max_wo_integrity[step][dimension]
                data.extend([[ansid, step+'_'+dimension, H1, an[3][step][dimension]/dmax]
                            for an in ranker1_ini if an[0] == ansid])
                R1.extend([an[3][step][dimension]/dmax for an in ranker1_ini if an[0] == ansid])
                data.extend([[ansid, step+'_'+dimension, H2, an[3][step][dimension]/dmax]
                            for an in ranker2_ini if an[0] == ansid])
                R2.extend([an[3][step][dimension]/dmax for an in ranker2_ini if an[0] == ansid])

        r, p_value = pearsonr(R1, R2)
        if display: print(H1 + ' and ' + H2 + ' in ' + ansid + '_ rating consistency: ', r, p_value)
        P.append(p_value)
        R.append(r)
    df = pd.DataFrame(data, columns = ["AnswerID", "Dimension", "Rater", "Score"])

    # ICC(3,k)
    icc = pg.intraclass_corr(
        data = df,
        targets = "AnswerID",
        raters = "Rater",
        ratings = "Score"
    )
    icc3k = icc[icc['Type'] == 'ICC3k'][['ICC','CI95%']]
    if display: 
        print('\n' + H1 + ' and ' + H2 + ' in ' + H1_FS + '_ the average, maximum, and minimum rating consistency for the ten responses are, respectively:', sum(R) / len(R), max(R), min(R))
        print(H1 + ' and ' + H2 + ' in ' + H1_FS + '_ the ICC consistency for the ten responses is:', icc3k['ICC'].values[0])
    
    return R, P, icc3k['ICC'].values[0]



def find_human_pairs(allocation, fs_order = None):
    '''
    Return the human rater IDs for all scenarios in the format: (H1_ID, H1_FS, H2_ID, H2_FS).
    '''
    fs_to_humans = defaultdict(list)
    for human, fs_list in allocation.items():
        for fs in fs_list:
            fs_to_humans[fs].append(human)
    if fs_order is None:
        fs_order = sorted(fs_to_humans.keys())
    
    pairs = []
    for fs in fs_order:
        humans = fs_to_humans.get(fs, [])
        assert len(humans) == 2
        for h1, h2 in combinations(humans, 2):
            pairs.append((h1, fs, h2, fs))

    return pairs



def calcu(PScore, threshold, human_allocation, condition):
    result = find_human_pairs(human_allocation, fs_order)
    all_P, ICC = [], []
    for r in result:
        P, _, icc = corr(PScore, r[0], r[1], r[2], r[3])
        all_P.append(P)
        ICC.append(icc)
    proportion = np.array(all_P) > threshold
    print('Condition', condition, '  Threshold: ', threshold)
    for i, fs in enumerate(fs_order):
        print('FS_ID: ', fs, ', Human_ID: ', result[i][0], ' and ', result[i][2], ', The proportion of the ten responses with consistency exceeding the threshold is:', np.mean(proportion[i]))
    print('The average proportion is:', np.mean(proportion))
    print('Average PCC: ', np.mean(all_P), ' max: ', np.max(all_P), ' min: ', np.min(all_P))
    print('Average ICC: ', np.mean(ICC), ' max: ', np.max(ICC), ' min: ', np.min(ICC))


''' Count the number of Blank entries in Step-1 or Step-3 responses '''
def get_blank(file_path: str) -> list:

    divergent_location = {
        'Step-1': ['H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12'],
        'Step-3': ['H56', 'H57', 'H58', 'H59', 'H60', 'H61', 'H62', 'H63']
    }
    PScore = []
    
    for f in list_excel_files(file_path): # For each scoring sheet
        file = os.path.join(file_path, f)
        sheets = get_excel_sheet_names(file)
        assert all('_' in item for item in sheets), file + 'There is an issue with the file or worksheet!'
        
        for sheet in sheets:
            df = pd.read_excel(file, sheet_name = sheet, header = None)
            def get_value(cell):
                row_idx, col_idx = coordinate_to_tuple(cell)
                return df.iat[row_idx - 1, col_idx - 1]
            TScore = [sheet]
            for step in ['Step-1', 'Step-3']:
                i = 0
                for li in divergent_location[step]:
                    if get_value(li) == 'B — No Response Provided': i += 1
                TScore.append(i)
            PScore.append(TScore)

    return PScore


def check_blank(blank_score: list):
    '''
    Check if any rating sheets have missing entries.
    '''
    ans_lst = sort_scores(set([an[0]for an in blank_score]))
    p = 0
    assert len(ans_lst) == 100, 'The number of responses is not 100!'
    for each_ans in ans_lst:
        evaluations = [an for an in blank_score if an[0] == each_ans]
        assert len(evaluations) == 2, each_ans + ' Fewer than two raters provided scores!'
        for step_num in [1, 2]:
            if evaluations[0][step_num] != evaluations[1][step_num]:
                p = 1
                print('Step-' + ('1' if step_num == 1 else '3') + ' of ' + each_ans + ': The number of blanks from the two raters is inconsistent!')
    if p == 0: print('All rating sheets have no missing entries.')

    def deduplicate_blank_score(blank_score: list):
        seen = set()
        unique_records = []
        for record in blank_score:
            ans_id = record[0]
            if ans_id not in seen:
                seen.add(ans_id)
                unique_records.append(record)
        assert len(unique_records) == 100, 'The number of responses after deduplication is not 100!'
        return unique_records

    return deduplicate_blank_score(blank_score)


def write_blank(blank_score_T, blank_score_S, save_path, save_name):
    def aggregate_blank(blank_score):
        """Return {model_id: {'step1': [...], 'step3': [...]}}"""
        data = defaultdict(lambda: {'step1': [], 'step3': []})
        for (ans_id, s1, s3) in blank_score:
            model_id = ans_id.split('_')[0]   # Extract A01, A02 ...
            data[model_id]['step1'].append(s1)
            data[model_id]['step3'].append(s3)
        return data
    
    agg_T, agg_S = aggregate_blank(blank_score_T), aggregate_blank(blank_score_S)
    all_models_T = sorted(set(list(agg_T.keys())))
    all_models_S = sorted(set(list(agg_S.keys())))

    # ---- Construct TRMAC Table ----
    trmac_data = {
        model_name[model[1:]]: [
            sum(agg_T[model]['step1']) / len(agg_T[model]['step1']) if agg_T[model]['step1'] else 0,
            sum(agg_T[model]['step3']) / len(agg_T[model]['step3']) if agg_T[model]['step3'] else 0
        ]
        for model in all_models_T
    }
    trmac_df = pd.DataFrame(trmac_data, index = ["Step-1-blank", "Step-3-blank"])
    
    # ---- Construct Single Table ----
    single_data = {
        model_name[model[1:]]: [
            sum(agg_S[model]['step1']) / len(agg_S[model]['step1']) if agg_S[model]['step1'] else 0,
            sum(agg_S[model]['step3']) / len(agg_S[model]['step3']) if agg_S[model]['step3'] else 0
        ]
        for model in all_models_S
    }
    single_df = pd.DataFrame(single_data, index = ["Step-1-blank", "Step-3-blank"])
    
    # ---- Saved to Excel ----
    save_file = os.path.join(save_path, save_name)
    with pd.ExcelWriter(save_file, engine="openpyxl") as writer:
        trmac_df.to_excel(writer, sheet_name="Result", startrow=0)
        single_df.to_excel(writer, sheet_name="Result", startrow=len(trmac_df)+3)
    
    print(f"Results saved to {save_file}")


''' Set the future scenario numbers assigned to each human expert '''
human_allocation_A = {
    'H01': ['FS1', 'FS10'],
    'H02': ['FS1', 'FS5', 'FS10'],
    'H03': ['FS2'],
    'H04': ['FS2'],
    'H05': ['FS3', 'FS8'],
    'H06': ['FS3', 'FS4', 'FS6', 'FS8'],
    'H07': ['FS5'],
    'H08': ['FS4'],
    'H09': ['FS7'],
    'H10': ['FS7'],
    'H11': ['FS6'],
    'H12': ['FS9'],
    'H13': ['FS9'],
}

human_allocation_B = {
    'H02': ['FS9'],
    'H06': ['FS1', 'FS3', 'FS5', 'FS7'],
    'H09': ['FS9'],
    'H11': ['FS1', 'FS8'],
    'H12': ['FS2', 'FS3', 'FS5', 'FS10'],
    'H13': ['FS2', 'FS4', 'FS6', 'FS8'],
    'H14': ['FS4', 'FS6'],
    'H15': ['FS7', 'FS10']
}

human_allocation_All = {
    'H01': ['FS1_A', 'FS10_A'],
    'H02': ['FS1_A', 'FS5_A', 'FS10_A', 'FS9_B'],
    'H03': ['FS2_A'],
    'H04': ['FS2_A'],
    'H05': ['FS3_A', 'FS8_A'],
    'H06': ['FS3_A', 'FS4_A', 'FS6_A', 'FS8_A', 'FS1_B', 'FS3_B', 'FS5_B', 'FS7_B'],
    'H07': ['FS5_A'],
    'H08': ['FS4_A'],
    'H09': ['FS7_A', 'FS9_B'],
    'H10': ['FS7_A'],
    'H11': ['FS6_A', 'FS1_B', 'FS8_B'],
    'H12': ['FS9_A', 'FS2_B', 'FS3_B', 'FS5_B', 'FS10_B'],
    'H13': ['FS9_A', 'FS2_B', 'FS4_B', 'FS6_B', 'FS8_B'],
    'H14': ['FS4_B', 'FS7_B'],
    'H15': ['FS7_B', 'FS10_B']
}


if __name__ == '__main__':

    # a = find_human_pairs(human_allocation_B, fs_order)
    # for i in a:
    #     print(i)

    # blank_score_T = get_blank(os.path.join('Human Evaluation', 'DATA_ACC'))
    # blank_score_S = get_blank(os.path.join('Human Evaluation', 'DATA_BCC'))
    # blank_score_T = check_blank(blank_score_T)
    # blank_score_S = check_blank(blank_score_S)
    # write_blank(blank_score_T, blank_score_S, 'Results', 'blank_num.xlsx')

    # for condition in ['A', 'B']:
    #     for ppp in ['', 'CC']:
    #         if ppp == '': print('The data before third-party calibration is as follows:')
    #         elif ppp == 'CC': print('The data after third-party calibration is as follows:')
    #         raw_scores = get_original_scores(os.path.join('Human Evaluation', 'DATA_' + condition + ppp))
    #         # check(raw_scores, human_allocation_A)
    #         scores_after_sort = sort_scores(raw_scores)
    #         calcu(scores_after_sort, 0.65, human_allocation_A if condition == 'A' else human_allocation_B, condition)
    #         print()
    #     print('=================================================\n')


    ''' TeaMAC '''
    raw_scores = get_original_scores(os.path.join('Human Evaluation', 'DATA_A'))
    raw_scores_C = get_original_scores(os.path.join('Human Evaluation', 'DATA_ACC'))
    check(raw_scores, human_allocation_A)
    print()
    scores_after_sort = sort_scores(raw_scores)
    scores_after_sort_C = sort_scores(raw_scores_C)

    # corr(scores_after_sort, 'H01', 'FS1', 'H02', 'FS1', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H01', 'FS1', 'H02', 'FS1', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H03', 'FS2', 'H04', 'FS2', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H03', 'FS2', 'H04', 'FS2', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H05', 'FS3', 'H06', 'FS3', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H05', 'FS3', 'H06', 'FS3', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H08', 'FS4', 'H06', 'FS4', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H08', 'FS4', 'H06', 'FS4', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H02', 'FS5', 'H07', 'FS5', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H02', 'FS5', 'H07', 'FS5', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H06', 'FS6', 'H11', 'FS6', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H06', 'FS6', 'H11', 'FS6', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H09', 'FS7', 'H10', 'FS7', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H09', 'FS7', 'H10', 'FS7', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H06', 'FS8', 'H05', 'FS8', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H06', 'FS8', 'H05', 'FS8', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H12', 'FS9', 'H13', 'FS9', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H12', 'FS9', 'H13', 'FS9', display = 1)
    # print('================================================\n')
    corr(scores_after_sort, 'H01', 'FS10', 'H02', 'FS10', display = 1)
    print('================================================\n')
    corr(scores_after_sort_C, 'H01', 'FS10', 'H02', 'FS10', display = 1)
    print('================================================\n')


    ''' Single Agent '''
    # raw_scores = get_original_scores(os.path.join('Human Evaluation', 'DATA_B'))
    # raw_scores_C = get_original_scores(os.path.join('Human Evaluation', 'DATA_BCC'))
    # check(raw_scores, human_allocation_B)
    # print()
    # scores_after_sort = sort_scores(raw_scores)
    # scores_after_sort_C = sort_scores(raw_scores_C)
    # corr(scores_after_sort, 'H06', 'FS1', 'H11', 'FS1', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H06', 'FS1', 'H11', 'FS1', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H12', 'FS2', 'H13', 'FS2', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H12', 'FS2', 'H13', 'FS2', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H06', 'FS3', 'H12', 'FS3', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H06', 'FS3', 'H12', 'FS3', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H14', 'FS4', 'H13', 'FS4', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H14', 'FS4', 'H13', 'FS4', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H12', 'FS5', 'H06', 'FS5', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H12', 'FS5', 'H06', 'FS5', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H13', 'FS6', 'H14', 'FS6', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H13', 'FS6', 'H14', 'FS6', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H15', 'FS7', 'H06', 'FS7', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H15', 'FS7', 'H06', 'FS7', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H11', 'FS8', 'H13', 'FS8', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H11', 'FS8', 'H13', 'FS8', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H02', 'FS9', 'H09', 'FS9', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H02', 'FS9', 'H09', 'FS9', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort, 'H15', 'FS10', 'H12', 'FS10', display = 1)
    # print('================================================\n')
    # corr(scores_after_sort_C, 'H15', 'FS10', 'H12', 'FS10', display = 1)
    # print('================================================\n')


    # Final_Scores = sort_scores(pairwise_average(raw_scores))
    # print(len(Final_Scores))