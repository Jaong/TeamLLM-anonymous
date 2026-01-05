import json
import time
from openai import OpenAI

tem_lst = {
    "Co-Ordinator": 0.5,
    "Plant": 0.8,
    "Monitor Evaluator": 0.4,
    "Implementer": 0.4
}

Single_Agent_temperature = 0.6

with open("config.json", 'r', encoding='utf-8') as f:
    config = json.load(f)

def process_stream(completion):
    reasoning_content = ""  # Full reasoning process
    answer_content = ""  # Full reply
    is_answering = False  # Whether the reply phase has started

    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            # if not is_answering:
            #     print(delta.reasoning_content, end="", flush=True)
            reasoning_content += delta.reasoning_content
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                # print("\n" + "=" * 20 + "Full reply" + "=" * 20 + "\n")
                is_answering = True
            # print(delta.content, end="", flush=True)
            answer_content += delta.content

    return reasoning_content, answer_content
    
class Agent():
    def __init__(self, model_type, model_name, agent_name):
        self.Model_type = model_type
        self.Model_name = model_name
        self.Agent_name = agent_name
        if self.Agent_name is not None and self.Agent_name in tem_lst:
            self.temperature = tem_lst[agent_name]
        else:
            self.temperature = Single_Agent_temperature
        self.memory_lst = []

    def set_meta_prompt(self, meta_prompt):
        self.memory_lst.append({"role": "system", "content": f"{meta_prompt}"})

    def add_assistant(self, content):
        self.memory_lst.append({"role": "assistant", "content": f"{content}"})

    def add_user(self, content):
        self.memory_lst.append({"role": "user", "content": f"{content}"})
    
    def ask(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


class OpenAI_Agent(Agent):
    def __init__(self, model_type, model_name, agent_name = None):
        super().__init__(model_type, model_name, agent_name)

        try:
            self.client = OpenAI(api_key = config['api_key'], base_url = "Your url here.")
        except KeyError:
            raise ValueError(f"Unknown Model_type: {self.Model_type}")
    
    def ask(self):
        try:
            kwargs = {}
            if self.Model_type == 'Meta':
                kwargs['max_tokens'] = 4096
            completion = self.client.chat.completions.create(
                model = self.Model_name,
                messages = self.memory_lst,
                temperature = self.temperature,
                **kwargs
            )
            result = completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Error with model {self.Model_name}: {e}")
            time.sleep(1)
            return self.ask()
        

class Qwen_thinking_Agent(OpenAI_Agent):
    def __init__(self, model_type, model_name, agent_name = None):
        super().__init__(model_type, model_name, agent_name)
    
    def ask(self):
        try:
            completion = self.client.chat.completions.create(
                model = self.Model_name,
                messages = self.memory_lst,
                extra_body={
                    "enable_thinking": True,
                    # "thinking_budget": 50 # Maximum number of tokens for the model's internal reasoning process
                    }, 
                stream = True,
                temperature = self.temperature
            )
            result = process_stream(completion)[1]
            return result
        except Exception as e:
            print(f"Error with model {self.Model_name}: {e}")
            time.sleep(1)
            return self.ask()
