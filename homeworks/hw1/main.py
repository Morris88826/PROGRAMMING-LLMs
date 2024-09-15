import os
import re   
import time
import json
import argparse
import subprocess
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

def init_api(model_type):
    load_dotenv()
    if model_type == "openai":
        client = OpenAI(api_key=os.environ["OPENAI_KEY"])
    elif model_type == "gemini":
        genai.configure(api_key=os.environ["GEMINI_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        client = model
    else:
        client = None
    return client

def verify_xbin(xbin_path):
    # Command to run
    verify_command = f"./run.sh -x run_pov {xbin_path} filein_harness"
    # Running the command and capturing the output
    cmd_return = subprocess.run(verify_command, shell=True, capture_output=True, text=True)
    return ("ERROR: AddressSanitizer: global-buffer-overflow" in cmd_return.stderr)

def verify_xdiff(xbin_path, xdiff_path):
    c1 = f"./run.sh -x build {xdiff_path} samples"
    c1_return = subprocess.run(c1, shell=True, capture_output=True, text=True)
    failed = ("error" in c1_return.stderr.lower())
    if failed:
        print("Failed to build the patched binary.")
        return False
    c2 = f"./run.sh -x run_pov {xbin_path} filein_harness"
    c2_return = subprocess.run(c2, shell=True, capture_output=True, text=True)
    f1 = ("error" in c2_return.stderr.lower())

    c3 = f"./run.sh -x run_tests"
    c3_return = subprocess.run(c3, shell=True, capture_output=True, text=True)
    f2 = ("error" in c3_return.stderr.lower())
    return False if f1 or f2 else True

def call_openai(prompt, client: OpenAI, deterministic=False):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        seed=0 if deterministic else None,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that detects vulnerabilities in C code."},
            {"role": "user", "content": prompt}
        ]
    )
    message = response.choices[0].message.content
    usage = response.usage
    metadata = {
        "token_count": usage.total_tokens,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens
    }
    return message, metadata

def call_gemini(prompt, client: genai.GenerativeModel):
    response = client.generate_content([prompt])
    message = response.text
    usage = response.usage_metadata
    metadata = {
        "token_count": usage.total_token_count,
        "prompt_tokens": usage.prompt_token_count,
        "completion_tokens": usage.candidates_token_count
    }
    return message, metadata

def call_local_llama3(prompt):
    dict_item = {
        "model": "llama3.1",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    json_data = json.dumps(dict_item)
    llama3_command = f'curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d \'{json_data}\''
    cmd_return = subprocess.run(llama3_command, shell=True, capture_output=True, text=True)

    response = json.loads(cmd_return.stdout)
    message = response["choices"][0]["message"]["content"]
    usage = response['usage']
    metadata = {
        "token_count": usage['total_tokens'],
        "prompt_tokens": usage['prompt_tokens'],
        "completion_tokens": usage['completion_tokens']
    }
    return message, metadata

def post_process(message, is_json=False):
    # Post-process the result
    message = message.strip('```').split("\n")[1:-1]
    message = "\n".join(message)

    if is_json:
        # Regular expression to extract JSON object
        json_pattern = r'({.*})'
        # Search for the JSON object
        match = re.search(json_pattern, message, re.DOTALL)
        if match:
            json_str = match.group(1)  # Extract the matched JSON string
            # Convert the string to a JSON object
            message = json.loads(json_str)
        else:
            print("No JSON object found.")
            raise ValueError("Failed to parse the result.")
    return message

def reset_state():
    r1 = "git -C src/samples reset --hard HEAD"
    subprocess.run(r1, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r2 = "./run.sh -x build"
    subprocess.run(r2, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate x.bin and x.diff files.')
    parser.add_argument('--src', type=str, default="./src/samples/mock_vp.c", help='Path to the source file.')
    parser.add_argument('--harness', type=str, default="./src/test/filein_harness.c", help='Path to the harness file.')
    parser.add_argument('--model', type=str, default="openai", choices=["openai", "gemini", "local_llama3"], help='Model to use for generating the x.bin and x.diff files.')
    parser.add_argument('--max_limit', type=int, default=10, help='Maximum number of attempts to generate the x.bin file.')
    parser.add_argument('-d', '--deterministic', action='store_true', help='Use deterministic mode for the model.')
    
    args = parser.parse_args()

    src_path = args.src
    harness_path = args.harness
    xbin_path = "x.bin"
    xdiff_path = "x.diff"

    if not os.path.exists(src_path) or not os.path.exists(harness_path):
        raise ValueError("The source file or the harness file does not exist.")
    

    # Reset the state
    print("Resetting the state of the repository.")
    reset_state()

    # Read the content of the C source file
    with open(src_path, 'r') as file:
        c_code = file.read()
    with open(harness_path, 'r') as file:
        harness_code = file.read()

    # Initialize the API
    print("Using the model:", args.model)
    client = init_api(args.model)
    max_lmit = args.max_limit

    print("==========Generating x.bin==========")
    return_format = {
        'func_a': '<array of strings>',
        'func_b': '<int>'
    }

    input_prompt = (
        f"""
        Given the following C code:
        {c_code}
        
        Provide an example of input that would cause this code to fail.

        The response should ONLY return the input causing the failure, formatted as follows:
        ```
        {json.dumps(return_format)}
        ```
        """
    )

    success = False
    start_t = time.time()
    num_total_tokens = 0
    num_prompt_tokens = 0
    num_completion_tokens = 0
    for i in range(max_lmit):
        print(f"Attempt {i+1}")
        if args.model == "openai":
            message, metadata = call_openai(input_prompt, client, deterministic=args.deterministic)
        elif args.model == "gemini":
            message, metadata = call_gemini(input_prompt, client)
        else:
            message, metadata = call_local_llama3(input_prompt)

        # Update the token count
        num_total_tokens += metadata['token_count']
        num_prompt_tokens += metadata['prompt_tokens']
        num_completion_tokens += metadata['completion_tokens']

        try:
            result = post_process(message, is_json=True)
            result_str = ""
            for res in result['func_a']:
                if res == "":
                    continue
                res.strip()
                result_str += res + "\n"
            result_str += "\n"
            result_str += str(result['func_b'])

        except Exception as e:
            print(f"Failed to parse the message: {e}")
            # print(message)
            # raise NotImplementedError("Failed to parse the result.")
            continue
        
        # Write the result to the x.bin file
        with open(xbin_path, "w") as file:
            file.write(result_str)
    
        # Verify the generated x.bin
        if verify_xbin(xbin_path):
            success = True
            print("Successfully generated the x.bin that triggers the vulnerability.")
            break

    assert success, "Failed to generate x.bin that triggers the vulnerability, please try again or increase the max_limit."
    
    print('Token count:', num_total_tokens)
    print('Prompt Tokens:', num_prompt_tokens)
    print('Completion Tokens:', num_completion_tokens)
    print(f"Time taken (s): {(time.time() - start_t): .2f}")

    print("==========Generating x.diff==========")

    input_prompt = (
        f"""
        Given the following C code (mock_vp.c):
        {c_code}

        And the following harness code (filein_harness.c) of how the c_code is used:
        {harness_code}
        
        Identify all potential vulnerabilities and issues in the code. These may include, but are not limited to:
        - Buffer overflows
        - Memory leaks
        - Misuse of pointers
        - Input validation issues
        - Concurrency or threading issues (if applicable)

        Provide MINIMAL changes to modify the c_code to ONLY correct these vulnerabilities while preserving the original functionality of the code. The following constraints must be met:
        - The code should run without runtime errors, such as segmentation faults or memory access violations, regardless of input.
        - The functionality of the code must remain intact, meaning that it should still achieve its intended purpose after the changes.
        
        The response must return ONLY the improved `improved_c_code` as follows:
        ```
        <modified_c_code>
        ```
        DO NOT include any additional information or explaination in the response. Just wrap the code within ``` and return it.
        """
    )

    success = False
    start_t = time.time()
    num_total_tokens = 0
    num_prompt_tokens = 0
    num_completion_tokens = 0
    for i in range(max_lmit):
        print(f"Attempt {i+1}")
        reset_state()

        if args.model == "openai":
            message, metadata = call_openai(input_prompt, client, deterministic=args.deterministic)
        elif args.model == "gemini":
            message, metadata = call_gemini(input_prompt, client)
        else:
            message, metadata = call_local_llama3(input_prompt)

        # Update the token count
        num_total_tokens += metadata['token_count']
        num_prompt_tokens += metadata['prompt_tokens']
        num_completion_tokens += metadata['completion_tokens']

        try:
            result = post_process(message)
            with open('./tmp.c', "w") as file:
                file.write(result)

            # generate the diff
            diff_command = f"diff -u {src_path} ./tmp.c"
            diff_return = subprocess.run(diff_command, shell=True, capture_output=True, text=True)
            diff_return = diff_return.stdout.strip()
            
            diff_return = diff_return.split('\n')[2:]
            diff_return = '\n'.join(diff_return)
            diff_return = "--- a/mock_vp.c\n+++ b/mock_vp.c\n" + diff_return
            diff_return += "\n"

            # remove the temporary file
            os.remove('./tmp.c')
            
            with open(xdiff_path, "w") as file:
                file.write(diff_return)
        except Exception as e:
            print(f"Failed to parse the message: {e}")
            # print(message)
            # raise NotImplementedError("Failed to parse the result.")
            continue

        # Verify the generated x.diff
        if verify_xdiff(xbin_path, xdiff_path):
            success = True
            print("Successfully generated the x.diff that fixes the vulnerability.")
            break

    assert success, "Failed to generate x.diff that fixes the vulnerability, please try again or increase the max_limit."

    print('Token count:', num_total_tokens)
    print('Prompt Tokens:', num_prompt_tokens)
    print('Completion Tokens:', num_completion_tokens)
    print(f"Time taken (s): {(time.time() - start_t): .2f}")
    print("==========Completed==========")