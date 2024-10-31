import os
import re
import enum
from dotenv import load_dotenv
from libs.llm import LLM
from libs.template import overview_template

class Task(enum.Enum):
    SEND_EMAIL = 0,
    READ_PDF = 1,
    SCHEDULE_MEETING = 2,
    SEARCH_INTERNET = 3
    ASK_QUESTION = 4


if __name__ == "__main__":
    print(f"I'm your email assistant, Monica.")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    llm = LLM(api_key)

    while True:
        try:
            # user_input = input("Please enter a new task: ")
            user_input = "I want to schedule a meeting."

            # determine the task
            prompt = overview_template.substitute(
                tasks=[task.name for task in Task],
                prompt=user_input
            )            
            llm_output = llm.invoke_global_llm(prompt)
            # Parse the output to determine the task
            regex = (
                r"Task\s*:[\s]*(.*)"
            )
            task_match = re.search(regex, llm_output, re.DOTALL) # re.DOTALL allows . to match newline
            if task_match:
                try:
                    task = Task[task_match.group(1).strip()]
                    if task == Task.SEND_EMAIL:
                        raise NotImplementedError("Please implement the email assistant.")
                    elif task == Task.READ_PDF:
                        raise NotImplementedError("Please implement the PDF assistant.")
                    elif task == Task.SCHEDULE_MEETING:
                        raise NotImplementedError("Please implement the meeting assistant.")
                    elif task == Task.SEARCH_INTERNET:
                        raise NotImplementedError("Please implement the search assistant.")
                    elif task == Task.ASK_QUESTION:
                        raise NotImplementedError("Please implement the question assistant.")
                except KeyError:
                    raise ValueError("Task not found in the LLM output.")
            else:
                raise ValueError("LLM output does not conform to the expected format.")

        except KeyboardInterrupt:
                print("\nExiting.\n")
                break
