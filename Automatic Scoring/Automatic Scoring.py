import json
import os
from ..Experiments.agents import OpenAI_Agent

responses_lst_A = [f"A{i:02d}_FS{j}" for i in range(1, 11) for j in range(1, 11)]
responses_lst_B = [f"B{i:02d}_FS{j}" for i in range(1, 11) for j in range(1, 11)]
responses_lst_C = [f"C{i:02d}_FS{j}" for i in range(1, 11) for j in range(1, 11)]
responses_lst_D = [f"D{i:02d}_FS{j}" for i in range(1, 11) for j in range(1, 11)]
total_responses_lst = responses_lst_A + responses_lst_B + responses_lst_C + responses_lst_D

def collect_step_dimensions(step_num, type):

    if type in [1, 2]:
        step_config = json.load(open(r"prompts\steps_config.json", 'r', encoding='utf-8'))["Step-" + str(step_num)]
    else:
        step_config = json.load(open(r"prompts\steps_config_few_shot.json", 'r', encoding='utf-8'))["Step-" + str(step_num)]
    Category_lst = "\n".join(json.load(open(r"prompts\category_lst.json", 'r', encoding='utf-8')))
    dimensions = ''
    for id, dim in enumerate(step_config['Step-rubrics'], start=1):
        dimensions += f"Dimension-{id}：{dim['dimension_name']}:\n{dim['dimension_rubrics'].replace('{Category_lst}', Category_lst)}\n\n"

    prompt = "Current step：{step_name}\nStep requirements：{step_description}\nThe evaluation dimensions and their corresponding criteria are as follows：\n\n{dimensions}".format(
        step_name = step_config['Step-Name'],
        step_description = step_config['Step-Description'],
        dimensions = dimensions
    )
    return prompt

def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {ans_id: {} for ans_id in total_responses_lst}

def Judge_Scoring(model_name, step_num, FS_num, response_content, type):
    
    Judge_LLM = OpenAI_Agent(model_type = 'OpenAI', model_name = model_name, agent_name = 'Judge_LLM')
    if type == 1:
        prompts = json.load(open(r"prompts\step_wise_evaluation.json", 'r', encoding='utf-8'))
        score_dimensions = collect_step_dimensions(step_num, type)
        output_template = json.load(open(r"prompts\Score_Template.json", 'r', encoding='utf-8'))["Step-" + str(step_num)]
    else:
        prompts = json.load(open(r"prompts\holistic_evaluation.json", 'r', encoding='utf-8'))
        score_dimensions = '\n'.join([collect_step_dimensions(step, type) for step in step_num])
        output_template = json.load(open(r"prompts\Score_Template.json", 'r', encoding='utf-8'))
    
    fs = json.load(open(r"Datesets\future_scenarios_en.json", 'r', encoding='utf-8'))['FS' + str(FS_num)]['text']
    user_prompt = prompts[1].format(step_num = 'Step-' + str(step_num), \
                                    future_scenario = fs, \
                                    response_content = response_content, \
                                    score_dimensions = score_dimensions,
                                    output_template = output_template)
    
    Judge_LLM.add_system(prompts[0])
    Judge_LLM.add_user(user_prompt)
    return Judge_LLM.ask()


def Evaluation(Judge_model_name, Pure_responses, step_num, type):

    file_name = "Evaluation" + str(type) + ".json"
    final_score = load_json_if_exists(file_name)
    
    if type == 1:
        for step in step_num:
            print('\n=======================================\n')
            for ans_id in Pure_responses:
                Judge_result = Judge_Scoring(Judge_model_name, step, int(ans_id.split("FS")[1]), Pure_responses[ans_id]['Step-' + str(step)], type)

                if Judge_result.startswith("```json") and Judge_result.endswith("```"):
                    Judge_result = Judge_result.strip("```json").strip("```")
                # print(Judge_result)
                
                Judge_result = json.loads(Judge_result)
                final_score[ans_id] = final_score[ans_id] | Judge_result
    else:
        print('\n=======================================\n')
        for ans_id in Pure_responses:
            Judge_result = Judge_Scoring(Judge_model_name, 1, int(ans_id.split("FS")[1]), Pure_responses[ans_id], type)

            if Judge_result.startswith("```json") and Judge_result.endswith("```"):
                Judge_result = Judge_result.strip("```json").strip("```")
            # print(Judge_result)
            
            Judge_result = json.loads(Judge_result)
            final_score[ans_id] = Judge_result

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(final_score, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    
    '''
    Pure_responses = {
            "A01_FS1": {'Step-1': ***, 'Step-2': ***, ...},
            ...
        }
    '''
    step_num = [1, 2, 3, 4, 5, 6]

    Pure_responses = json.load(open(r"jsons\pure_responses.json", 'r', encoding='utf-8'))

    # step-wise evaluation
    Evaluation('gpt-5', Pure_responses, step_num, type = 1)

    # Holistic evaluation
    Evaluation('gpt-5', Pure_responses, step_num, type = 2)

    # Holistic evaluation with few-shot
    Evaluation('gpt-5', Pure_responses, step_num, type = 3)