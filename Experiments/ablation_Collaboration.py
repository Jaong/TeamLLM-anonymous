import json
from ..Experiments.agents import OpenAI_Agent, Qwen_thinking_Agent

class GeneralExperiment():
    def __init__(self, Model_Type, Model_Name, Model_ID):
        self.FPSP_FS = self.load_json(r'Datasets\future_scenarios.json')
        self.FPSP_Steps = self.load_json(r'prompts\FPSP.json')
        self.prompts = self.load_json(r'prompts\prompts.json')
        self.step_answer = []
        self.Model_Type = Model_Type
        self.Model_Name = Model_Name
        self.Model_ID = Model_ID

    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def construct_step_prompt(self, step):
        FPSP_step = self.FPSP_Steps[step - 1]
        step_prompt = self.prompts['Task_Step'].format(
            Step_Number = FPSP_step['Step_Number'], Step_Name = FPSP_step['Step_Name'],\
            Step_Description = FPSP_step['Step_Description'], Step_Output = FPSP_step['Step_Output'])
        return step_prompt


class AblationCollaboration(GeneralExperiment):
    def __init__(self, Model_Type, Model_Name, Model_ID, rounds, display):
        super().__init__(Model_Type, Model_Name, Model_ID)
        self.NAME_LST = [
            'Member 1',
            'Member 2',
            'Member 3',
            'Member s4'
        ]
        self.agents = self.initialize_agents()
        self.rounds = rounds
        self.history = []
        self.step_his = []
        self.log_his = ['The following is the historical record up to now: ']
        
        self.display = display
        

    def initialize_agents(self):
        agents = []
        # for _ in range(4):
        for agent_name in self.NAME_LST:
            if self.Model_Type == 'Qwen_thinking':
                agents.append(Qwen_thinking_Agent(model_type = self.Model_Type, 
                                                model_name = self.Model_Name,
                                                agent_name = agent_name))
            else:
                agents.append(OpenAI_Agent(model_type = self.Model_Type, 
                                            model_name = self.Model_Name,
                                            agent_name = agent_name))
            
        # for dic in self.TeamRole:
        #     self.role_play.append(self.prompts["Team_Role_Setting"].format(Team_Role = dic['Agent_name'], \
        #         Role_Speciality = dic['Role_Speciality'], Role_Prompt = dic['Role_Prompt'], \
        #         Other_Menbers = '、'.join([item for item in self.NAME_LST if item != dic['Agent_name']])))

        return agents

    def construct_meta_prompt(self, FS_str, agent):
        # initial_system_prompt = self.prompts["FPSP_Overall_Description_Team"] + "\n\n" + \
        #     self.prompts["Future_Scenario"].format(Future_Scenario = self.FPSP_FS[FS_str]['text']) + "\n\n" + \
        #     self.prompts['Emphasis_Team']
        initial_system_prompt = self.prompts["ablation_prompt"].format(Agent_Name = agent.Agent_name, Other_Members = '、'.join([item for item in self.NAME_LST if item != agent.Agent_name])) + "\n\n" + \
            self.prompts["Future_Scenario"].format(Future_Scenario = self.FPSP_FS[FS_str]['text']) + "\n\n" + \
            self.prompts['Emphasis_Individual']
        return initial_system_prompt


    def update_step_history(self, agent, message, step_history):
        step_history.append(agent.Agent_name + ' says: ' + message)
        if self.display == 1: 
            print(step_history[-1])
        return step_history


    def run(self, step, FS_str):
        '''
        Arguments:
            Inputs:
                step: int, current step number from 0 to 6, with 0 indicating the initial phase
                FS_str: str, Future Scenario string, e.g., 'FS1', 'FS2', etc.
        '''
        print("============Running step -", step, '============\n')
        step_history = []

        if step == 0: 
            for agent in self.agents:
                agent.set_meta_prompt(self.construct_meta_prompt(FS_str, agent))
            for i, agent in enumerate(self.agents):
                agent.add_user(self.role_play[i] + '\n\n' + 'The following is the historical record up to now: ' + '\n'.join(step_history) + \
                               '\n\n' + 'Please note, this stage does not involve tasks, only introduce yourself. Keep your introduction within 120 words.')
                warm_up = agent.ask()
                agent.add_assistant(warm_up)
                step_history = self.update_step_history(agent, warm_up, step_history)
            
            self.history = step_history
            self.step_his.append(step_history.copy())
            
        else:
            step_prompt = super().construct_step_prompt(step)
            for i, agent in enumerate(self.agents):
                agent.add_user(step_prompt + '\n\n' + '\n\n'.join(self.log_his + step_history))
                ans = agent.ask()
                agent.add_assistant(ans)
                step_history = self.update_step_history(agent, ans, step_history)

            self.history += step_history
            self.step_his.append(step_history.copy())
            plain_answer = step_history[-1][step_history[-1].index('：') + 1:].strip()
            self.step_answer.append(plain_answer)
            self.log_his.append('The answer of Step-' + str(step) + ' is：\n' + plain_answer)

        if step == 6:
            return self.history, self.step_his, self.step_answer