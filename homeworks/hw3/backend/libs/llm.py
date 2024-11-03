import openai
import requests

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434/api/generate"):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json"
        }

    def generate(self, model, prompt, stream=False):
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(self.base_url, json=data, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")


class LLM:
    def __init__(self, OPENAI_API_KEY:str, local_llm_type="Ollama"):
        self.local_llm_type = local_llm_type
        if self.local_llm_type == "Ollama":
            self.local_client = OllamaClient()
        else: # use llama.cpp
            self.local_client = openai.Client(base_url="http://127.0.0.1:8080/v1", api_key="EMPTY")

        self.global_client = openai.Client(api_key=OPENAI_API_KEY)
    
    def invoke_global_llm(self, prompt:str) -> str:
        try:
            response = self.global_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a personal assistant."},
                    {"role": "user", "content": prompt}
                ],
            )
            output = response.choices[0].message.content
        except Exception as e:
            output = f"Exception: {e}"
        return output

    def invoke_local_llm(self, prompt:str) -> str:
        if self.local_llm_type == "Ollama":
            try:
                response = self.local_client.generate(model="llama3.2", prompt=prompt)
                output = response["response"]
            except Exception as e:
                output = f"Exception: {e}"
        else:
            stop = ['Observation:', 'Observation ']
            try:
                response = self.local_client.completions.create(
                    model="llama3.2",
                    prompt=prompt,
                    stop=stop,
                    max_tokens=200,
                )
                output = response.content
            except Exception as e:
                output = f"Exception: {e}"

        return output