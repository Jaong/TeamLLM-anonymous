from Extract_Scores import get_original_scores, sort_scores, check, pairwise_average, steps_dimensions_max_wo_integrity,\
        steps_dimensions_wo_integrity, steps_dimensions_max_with_integrity, steps_dimensions_with_integrity, \
        human_allocation_A, human_allocation_B, ScoreName_lst, model_name, fs_order
from openpyxl import Workbook
from scipy.stats import friedmanchisquare, rankdata
from scipy.stats import studentized_range
import math
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment, Border, Side, Font
import matplotlib.pyplot as plt
import os
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib.patches import FancyArrowPatch, Patch


def write_integrity(EScore: list) -> list:
    '''
    Combine the following fields in Step-2 of each scoring data: 'Condition Phrase', 'Stem & KVP', 'Purpose', and 'FS Parameters' into a single dimension called 'Integrity'.
    '''
    for an in EScore:
        data = an[3]['Step-2']
        inte = sum(data[a] for a in ['Condition Phrase', 'Stem & KVP', 'Purpose', 'FS Parameters'])

        new = {'Integrity': int(inte) if inte % 1 == 0 else inte}
        old = dict(list(data.items())[4:])
        an[3]['Step-2'] = {**new, **old}

    return EScore


def average_FS(EScore: list, model_lst: list, FS_lst: list) -> list:
    '''
    Compute the average score for each dimension across all future scenarios under the same condition.

    Returned data format:
        AScore = [
            [Model ID,
                {
                    'Step-1': {dimension1: score, dimension2: score, ...},
                    'Step-2': {...},
                    ...
                    'Step-6': {...},
                    'Overall': {...},
                    'Total Score': total_score_value
                }
            ],
            ...
        ]

    '''
    def clean(avg):
        if avg % 1 == 0.5:
            return avg
        elif avg % 1 == 0:
            return int(avg)
        else:
            return round(avg, 2)

    AScore = []
    for model in model_lst:
        CP, Sdic = [model], {}
        ans_lst = [tn for tn in EScore if tn[0].startswith(model)]
        assert len(ans_lst) == len(FS_lst), 'The number of responses for this model is incorrect!'
        for step in steps_dimensions_with_integrity:
            Odic = {}
            for dim in steps_dimensions_with_integrity[step]:
                avg = sum([tm[3][step][dim] for tm in ans_lst]) / len(ans_lst)
                Odic[dim] = clean(avg)
            Sdic[step] = Odic

        Sdic['Total Score'] = clean(sum([tm[3]['Total Score'] for tm in ans_lst]) / len(ans_lst))
        CP.append(Sdic)
        AScore.append(CP)

    return AScore
            

def ini(EScore: list) -> list:
    '''
    Return the lists of all sorted model IDs and future scenario IDs:

    Model IDs: ['A01', 'A02', 'A03', 'A04']

    Future Scenario IDs: ['FS1', 'FS2', 'FS3', 'FS4', 'FS5', 'FS6', 'FS7', 'FS8', 'FS9', 'FS10']
'''
    model_ID_lst = set([an[0].split('_')[0] for an in EScore])
    FS_lst = set([an[2].split('_')[0] for an in EScore])
    return sorted(model_ID_lst), sorted(FS_lst, key = lambda x: int(x[2:]))