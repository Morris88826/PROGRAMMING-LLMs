import os
import subprocess
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API Key not found. Please set the API Key in the .env file.")
genai.configure(api_key=os.environ["API_KEY"])

def verify_xbin(xbin_path):
    # Command to run
    verify_command = f"./run.sh -x run_pov {xbin_path} filein_harness"
    # Running the command and capturing the output
    cmd_return = subprocess.run(verify_command, shell=True, capture_output=True, text=True)
    return ("ERROR: AddressSanitizer: global-buffer-overflow" in cmd_return.stderr)

def generate_xbin(src_path, harness_path):
    # Read the content of the C source file
    with open(src_path, 'r') as file:
        c_code = file.read()

    with open(harness_path, 'r') as file:
        harness_code = file.read()

    # create the model
    model = genai.GenerativeModel("gemini-1.5-flash")

    input_prompt = (
        f"""
        Given the following C code:
        {c_code}

        And here is the code that harnesses the C code:
        {harness_code}

        Provide an example of input that would cause this code to fail.

        The input must contains the part for `func_a` and `func_b`, the two inputs must be seperated with an empty line.

        IMPORTANT: Return only the example input that causes the failure, with no explanations or comments.
        """
    )

    # Generate the content
    result = model.generate_content([input_prompt])

    result = result.text.strip().split('\n')[1:-1]
    result = '\n'.join(result)

    return result

def verify_xdiff(xbin_path, xdiff_path):
    c1 = f"./run.sh -x build {xdiff_path} samples"
    c1_return = subprocess.run(c1, shell=True, capture_output=True, text=True)
    if "error" in c1_return.stderr:
        print("Failed to build the patched binary.")
        return False
    c2 = f"./run.sh -x run_pov {xbin_path} filein_harness"
    cmd_return = subprocess.run(c2, shell=True, capture_output=True, text=True)

    return not ("error" in cmd_return.stderr.lower())

def generate_xdiff(src_path, harness_path):
    # Read the content of the C source file
    with open(src_path, 'r') as file:
        c_code = file.read()

    with open(harness_path, 'r') as file:
        harness_code = file.read()

    # create the model
    model = genai.GenerativeModel("gemini-1.5-flash")

    input_prompt = (
        f"""
        Given the following C code (mock_vp.c):
        {c_code}

        And here is the code that harnesses the C code:
        {harness_code}

        Identify all potential vulnerabilities and issues in the code. These may include, but are not limited to:
        - Buffer overflows
        - Memory leaks
        - Misuse of pointers
        - Input validation issues
        - Concurrency or threading issues (if applicable)

        Provide a solution to correct these vulnerabilities while preserving the original functionality of the code. The following constraints must be met:
        - The code should run without runtime errors, such as segmentation faults or memory access violations, regardless of input.
        - The functionality of the code must remain intact, meaning that it should still achieve its intended purpose after the changes.

        IMPORTANT: Do **not** include any explanations in the output. Return **only** the modified C code, without comments or explanations.
        """
    )
        

    result = model.generate_content([input_prompt])
    result = result.text.strip()

    result = result.split('\n')
    result = result[1:-1]
    result = '\n'.join(result)

    with open('./tmp.c', "w") as file:
        file.write(result)

    # generate the diff
    diff_command = f"diff -u {src_path} ./tmp.c"
    diff_return = subprocess.run(diff_command, shell=True, capture_output=True, text=True)
    diff_return = diff_return.stdout.strip()
    
    diff_return = diff_return.split('\n')[2:]
    diff_return = '\n'.join(diff_return)
    diff_return = "--- a/mock_vp.c\n+++ b/mock_vp.c\n" + diff_return

    # remove the temporary file
    os.remove('./tmp.c')

    return diff_return

def reset_state():
    r1 = "git -C src/samples reset --hard HEAD"
    subprocess.run(r1, shell=True)
    r2 = "./run.sh -x build"
    subprocess.run(r2, shell=True)

if __name__ == "__main__":
    src_path = "./src/samples/mock_vp.c"
    harness_path = "./src/test/filein_harness.c"
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"File not found: {src_path}")
    
    xbin_path = "x.bin"
    xdiff_path = "x.diff"

    # Reset state
    print("Resetting the state of the repository.")
    reset_state()
    
    print("=====================================")
    print("Generating x.bin:")
    ##### Generate the x.bin
    max_lmit = 10
    for i in range(max_lmit):
        print(f"Attempt {i+1}")
        result = generate_xbin(src_path, harness_path)
        with open(xbin_path, "w") as file:
            file.write(result)
    
        # Verify the generated x.bin
        if verify_xbin(xbin_path):
            print("Successfully generated the x.bin that triggers the vulnerability.")
            break
    if i == max_lmit - 1:
        print("Failed to generate x.bin that triggers the vulnerability.")

    ##### Generate the x.diff
    print("=====================================")
    print("Generating x.diff:")
    max_lmit = 10
    for i in range(max_lmit):
        print(f"Attempt {i+1}")
        reset_command = "git -C src/samples reset --hard HEAD"
        subprocess.run(reset_command, shell=True)
        result = generate_xdiff(src_path, harness_path)
        with open(xdiff_path, "w") as file:
            file.write(result)

        # Verify the generated x.diff
        if verify_xdiff(xbin_path, xdiff_path):
            print("Successfully generated the x.diff that fixes the vulnerability.")
            break
    if i == max_lmit - 1:
        print("Failed to generate x.diff that fixes the vulnerability.")