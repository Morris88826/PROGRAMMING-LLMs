import openai

class LLM:
    def __init__(self, OPENAI_API_KEY:str):
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
        stop = ['Observation:', 'Observation ']
        try:
            response = self.local_client.completions.create(
                model="llama3.1",
                prompt=prompt,
                stop=stop,
                max_tokens=200,
            )
            output = response.content
        except Exception as e:
            output = f"Exception: {e}"

        return output