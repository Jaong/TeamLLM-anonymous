from ablation_Collaboration import AblationCollaboration
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Process_Results.Write_Answers import complete_answer

def run_Ablation(Model_Type, Model_Name, Model_ID, fs, display = 0):
    name = 'D' + Model_ID + '_' + fs
    print('\n', "============   " + name + ' (' + Model_Name + ')' + '   ============\n')
    Team = AblationCollaboration(Model_Type, Model_Name, Model_ID, rounds = 1, display = display)

    for step in range(7):
        history = Team.run(step, fs)

    complete_answer(history, name, 'D' + Model_ID)

FS_lst = [f"FS{i}" for i in range(5, 6)]
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


for fs in FS_lst:
    for Model_Type, Model_Name, Model_ID in Model_lst[::-1]:
        run_Ablation(Model_Type, Model_Name, Model_ID, fs, display = 0)
