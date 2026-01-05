'''
This code file is used to run the code programs under two experimental conditions.
'''

from Collaboration import RoleplayCollaboration, Single
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Process_Results.Write_Answers import complete_answer, write_time, write_single_answer

def run_Team(Model_Type, Model_Name, Model_ID, display = 0):
    Time = [] # for storing time consumption for each future scenario
    for FS_str in FS_lst: # run through all future scenarios one by one
        name = 'A' + Model_ID + '_' + FS_str
        start_time = time.time()
        print('\n', "============   " + name + ' (' + Model_Name + ')' + '   ============\n')
        Team = RoleplayCollaboration(Model_Type, Model_Name, Model_ID, rounds = 1, display = display)

        for step in range(7):
            history = Team.run(step, FS_str)

        complete_answer(history, name, 'A' + Model_ID)
        end_time = time.time()
        Time.append(round(end_time - start_time, 6))
    
    write_time(Time, 'A' + Model_ID, FS_lst)
    

def run_Single(Model_Type, Model_Name, Model_ID):
    Time = []
    for FS_str in FS_lst:
        name = 'B' + Model_ID + '_' + FS_str
        start_time = time.time()
        print('\n', "============   " + name + ' (' + Model_Name + ')' + '   ============\n')
        Team = Single(Model_Type, Model_Name, Model_ID)

        for step in range(1, 7):
            step_answer = Team.run(step, FS_str)

        write_single_answer(step_answer, name, 'B' + Model_ID)
        end_time = time.time()
        Time.append(round(end_time - start_time, 6))
    
    write_time(Time, 'B' + Model_ID, FS_lst)


FS_lst = ['FS1', 'FS2', 'FS3', 'FS4', 'FS5', 'FS6', 'FS7', 'FS9', 'FS10'] # Future Scenarios list
# FS_lst = [f"FS{i}" for i in range(5, 6)]
Model_lst = [
    ['Qwen', 'qwen3-235b-a22b-instruct-2507', '01'],
    ['Qwen_thinking', 'qwen3-235b-a22b-thinking-2507', '02'],
    ['DeepSeek', 'deepseek-chat', '03'],
    ['DeepSeek', 'deepseek-reasoner', '04'],
    ['Moonshot', 'kimi-k2-0711-preview', '05'],
    ['Meta', 'llama-4-scout-17b-16e-instruct', '06'],
    ['OpenAI', 'gpt-4o', '07'],
    ['OpenAI', 'gpt-5', '08'],
    ['Anthropic', 'claude-opus-4-1-20250805', '09'],
    ['DeepMind', 'gemini-2.5-pro', '10']
]

# running TeaMAC Condition
for Model_Type, Model_Name, Model_ID in Model_lst:
    run_Team(Model_Type, Model_Name, Model_ID, display = 0)

# running Single-Agent Condition
# for Model_Type, Model_Name, Model_ID in Model_lst:
#     run_Single(Model_Type, Model_Name, Model_ID)