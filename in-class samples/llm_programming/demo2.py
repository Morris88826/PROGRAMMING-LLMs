import openai
client = openai.Client(
    base_url="http://127.0.0.1:11434/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="llama3.1",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print(f"answer: {response.choices[0].message.content}")
