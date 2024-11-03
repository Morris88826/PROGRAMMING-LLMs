import os
import time
import base64
import pandas as pd
from requests import HTTPError
from email.mime.text import MIMEText
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from libs.llm import LLM

def find_email(query: str) -> str:
    pass

def send_email(to_addr: str, subject: str, body: str):
    SCOPES = [
        "https://www.googleapis.com/auth/gmail.send"
    ]
    flow = InstalledAppFlow.from_client_secrets_file('./private/credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    service = build('gmail', 'v1', credentials=creds)
    message = MIMEText(body)
    message['to'] = to_addr
    message['subject'] = subject
    create_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
    try:
        message = (service.users().messages().send(userId="me", body=create_message).execute())
        print(F'sent message to {message} Message Id: {message["id"]}')
    except HTTPError as error:
        print(F'An error occurred: {error}')
        message = None


if __name__ == "__main__":
    print(f"I'm your email assistant, Monica.")
    load_dotenv()

    send_email("morris88826@gmail.com", "Hello", "Hello, this is a test email")
    # api_key = os.getenv("OPENAI_API_KEY")
    # llm = LLM(api_key)

    # while True:
    #     try:
    #         # user_input = input("Please enter a new task: ")
    #         user_input = "can you help me send an email?"
    #         tic = time.time()
    #         print(llm.invoke_global_llm(user_input))
    #         raise NotImplementedError("Please implement the email assistant.")
    #         latency = time.time() - tic
    #         print(f"\nLatency: {latency:.3f}s")
    #     except KeyboardInterrupt:
    #             print("\nExiting.\n")
    #             break
