import os
import requests
import time
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='Resume_GPT/Multi-Turn-QA-System/log/chatbot.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Azure OpenAI Configurations
AZURE_OPENAI_ENDPOINT = os.environ.get("AZUREAI_ENDPOINT_URL")
AZURE_OPENAI_API_KEY = os.environ.get("AZUREAI_API_KEY")


if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise ValueError("Please set the AZUREAI_ENDPOINT_URL and AZUREAI_API_KEY environment variables.")

headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_API_KEY,
}

class Config:
    SYSTEM_PROMPT = "You are a helpful assistant that helps users build their resume."
    TEMPERATURE = 0.7
    AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_ENDPOINT
    headers = headers

def gpt_generate_use_azure(prompt: str, model: str = 'gpt-4', max_tokens: int = 500) -> str:
    """
    Calls GPT-4 API via Azure OpenAI to generate a response.
    Implements a retry mechanism to handle transient failures.
    """
    # Retry mechanism parameters
    max_retries = 10       # Maximum number of retries
    retry_delay = 30       # Delay between retries (seconds)
    attempt = 0            # Current attempt count

    # Build the full API endpoint URL, including deployment name and API version
    endpoint = f"{Config.AZURE_OPENAI_ENDPOINT}"
    logging.debug(f"User prompt: {prompt}")
    while attempt <= max_retries:
        try:
            # Request payload
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": Config.SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": Config.TEMPERATURE,
                "max_tokens": max_tokens,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }

            response = requests.post(endpoint, headers=Config.headers, json=payload)
            response.raise_for_status()  # Raise HTTP errors
            response_data = response.json()

            if 'choices' in response_data and len(response_data['choices']) > 0:
                completion_content = response_data['choices'][0]['message']['content']
                message = completion_content.strip()
                logging.debug(f"GPT-4 response: {message}")
            else:
                logging.error("No choices found in the response.")
                return ""

            return message

        except requests.RequestException as e:
            attempt += 1
            if attempt <= max_retries:
                logging.warning(f"API request failed: {e}. Retrying in {retry_delay} seconds... (Attempt {attempt}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"API request failed after {max_retries} attempts: {e}.")
                return ""
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON response: {e}")
            return ""

# Updated resume template
resume_template = {
    "personal_info": {
        "name": None,
        "age": None,
        "gender": None,
        "phone_number": None,
        "email": None,
        "address": None,
        "linkedin": None,
        "github": None
    },
    "education_info": [  # List of dicts
        {
            "school": None,
            "major": None,
            "degree": None,
            "gpa": None,
            "education_time": None,
            "courses": [],
            "awards": []
        }
    ],
    "work_experience": [  # List of dicts
        {
            "company": None,
            "title": None,
            "location": None,
            "time": None,
            "details": []
        }
    ],
    "projects": [  # List of dicts
        {
            "name": None,
            "role": None,
            "time": None,
            "details": []
        }
    ],
    "skills": {  # Dict
        "Programming": None,
        "Languages": None,
        "Tools": None
    },
    "research_interests": None,  # String
    "research_experience": [  # List of dicts
        {
            "title": None,
            "supervisor": None,
            "time": None,
            "details": []
        }
    ],
    "publications": [],  # List of strings
    "awards": [],  # List of strings
}

def get_expected_keys_for_main_slot(main_slot):
    expected_keys = {
        "education_info": ["school", "major", "degree", "gpa", "education_time", "courses", "awards"],
        "work_experience": ["company", "title", "location", "time", "details"],
        "projects": ["name", "role", "time", "details"],
        "research_experience": ["title", "supervisor", "time", "details"],
        # No need for publications and awards as they are lists of strings
    }
    return expected_keys.get(main_slot, [])



def intent_recognition(user_input):
    prompt = f"""
Given the user's input: "{user_input}", determine whether the user's intent is resume building (return 1) or not (return 0). Only return the number 1 or 0.
"""
    response = gpt_generate_use_azure(prompt)
    # Try to extract the number from the response
    try:
        intent = int(response.strip())
        return intent
    except ValueError:
        logging.error(f"Failed to parse intent from GPT response: {response}")
        return 0

def parse_sub_parameters(main_slot, main_slot_value):
    if isinstance(resume_template[main_slot], list):
        # Check if it's a list of dicts or list of strings
        if resume_template[main_slot]:
            first_item = resume_template[main_slot][0]
            if isinstance(first_item, dict):
                # List of dicts
                sub_params_keys = list(first_item.keys())
                sub_params_type = 'list_of_dicts'
            else:
                # List of strings
                sub_params_keys = []
                sub_params_type = 'list_of_strings'
        else:
            # Empty list, need to define expected keys for list of dicts
            if main_slot in ["education_info", "work_experience", "projects", "research_experience"]:
                sub_params_keys = get_expected_keys_for_main_slot(main_slot)
                sub_params_type = 'list_of_dicts'
            else:
                sub_params_keys = []
                sub_params_type = 'list_of_strings'
    elif isinstance(resume_template[main_slot], dict):
        sub_params_keys = list(resume_template[main_slot].keys())
        sub_params_type = 'dict'
    else:
        # Main slot does not have sub-parameters (string)
        sub_params_keys = []
        sub_params_type = 'string'

    # Build a prompt to parse sub_parameters from main_slot_value
    if sub_params_type == 'list_of_dicts' or sub_params_type == 'dict':
        prompt = f"""
Given the main slot "{main_slot}" with value "{main_slot_value}", extract the sub-parameters and their values.

Sub-parameters are:

{sub_params_keys}

Return a JSON object with sub-parameter names as keys and their extracted values.

Do not include any code fences or markdown formatting in your response. Only provide the JSON object.
"""
    else:
        # For list of strings or simple strings, no sub-parameters
        return main_slot_value.strip()

    response = gpt_generate_use_azure(prompt)
    try:
        # Clean up the response to remove code fences and language tags
        response_cleaned = response.strip()
        # Remove code fences if they exist
        if response_cleaned.startswith("```"):
            response_cleaned = response_cleaned.strip('`')
            # Remove possible language tag like "json"
            if response_cleaned.startswith("json"):
                response_cleaned = response_cleaned[len("json"):].strip()
        sub_params = json.loads(response_cleaned)
        logging.debug(f"Parsed sub-parameters for {main_slot}: {sub_params}")
        return sub_params
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse sub-parameters from GPT response: {response}")
        logging.error(f"JSONDecodeError: {e}")
        return {}




def generate_followup_questions(main_slot, missing_sub_parameters):
    # Build a prompt to generate follow-up questions
    prompt = f"""
You are helping the user fill in their resume

The user has not provided the following information: {missing_sub_parameters}.

Generate a natural language question to ask the user for this information.
"""
    response = gpt_generate_use_azure(prompt)
    question = response.strip()
    logging.debug(f"Generated follow-up question for {main_slot}: {question}")
    return question


def main():
    # Initialize the resume data
    resume_data = {key: None for key in resume_template.keys()}
    raw_inputs = {key: "" for key in resume_template.keys()}  # For storing raw user inputs

    # Flags to keep track of completion
    main_slots_completed = {key: False for key in resume_data.keys()}
    conversation_active = True

    print("Bot: Hello! I can help you build your resume. Let's get started.")

    main_slots = list(resume_template.keys())
    current_main_slot_index = 0

    while conversation_active and current_main_slot_index < len(main_slots):
        main_slot = main_slots[current_main_slot_index]
        # If main_slot value is None, ask the user for it
        if resume_data[main_slot] is None:
            if isinstance(resume_template[main_slot], list):
                # Determine if it's a list of dicts or list of strings
                if resume_template[main_slot]:
                    first_item = resume_template[main_slot][0]
                    if isinstance(first_item, dict):
                        list_type = 'list_of_dicts'
                    else:
                        list_type = 'list_of_strings'
                else:
                    if main_slot in ["education_info", "work_experience", "projects", "research_experience"]:
                        list_type = 'list_of_dicts'
                    else:
                        list_type = 'list_of_strings'

                resume_data[main_slot] = []
                adding_items = True
                while adding_items:
                    print(f"Bot: Please provide details for {main_slot.replace('_', ' ')} (or type 'done' to finish).")
                    user_input = input("User: ")
                    if user_input.lower() in ["done", "exit", "quit"]:
                        adding_items = False
                        if user_input.lower() in ["exit", "quit"]:
                            conversation_active = False
                        break

                    if list_type == 'list_of_dicts':
                        # Parse sub-parameters for the item
                        item_sub_params = parse_sub_parameters(main_slot, user_input)
                        # Check for missing sub-parameters for this entry
                        sub_params_keys = get_expected_keys_for_main_slot(main_slot)
                        missing_sub_params = [key for key in sub_params_keys if key not in item_sub_params or not item_sub_params[key]]
                        while missing_sub_params:
                            # Generate follow-up question
                            question = generate_followup_questions(main_slot, missing_sub_params)
                            print(f"Bot: {question}")
                            followup_input = input("User: ")
                            # Append follow-up input to the user's initial input
                            user_input += " " + followup_input
                            # Re-parse sub-parameters
                            item_sub_params = parse_sub_parameters(main_slot, user_input)
                            missing_sub_params = [key for key in sub_params_keys if key not in item_sub_params or not item_sub_params[key]]
                        resume_data[main_slot].append(item_sub_params)
                    else:
                        # List of strings
                        resume_data[main_slot].append(user_input.strip())
            elif isinstance(resume_template[main_slot], dict):
                # Handle dict-type main slots
                print(f"Bot: Please provide your {main_slot.replace('_', ' ')}.")
                user_input = input("User: ")
                logging.debug(f"User input for {main_slot}: {user_input}")
                if user_input.lower() in ["exit", "quit"]:
                    print("Bot: Thank you for using the Resume Builder. Goodbye!")
                    break
                raw_inputs[main_slot] += " " + user_input  # Store the raw input
            else:
                # Handle main slots without sub-parameters
                print(f"Bot: Please provide your {main_slot.replace('_', ' ')}.")
                user_input = input("User: ")
                resume_data[main_slot] = user_input.strip()
                main_slots_completed[main_slot] = True
                current_main_slot_index += 1
                continue
        else:
            # Main slot already has data
            if isinstance(resume_template[main_slot], dict):
                user_input = input("User: ")
                raw_inputs[main_slot] += " " + user_input
            else:
                # For list-type or other main slots, move to the next
                current_main_slot_index += 1
                continue

        # Parsing sub-parameters for dict-type main slots
        if isinstance(resume_template[main_slot], dict):
            sub_params = parse_sub_parameters(main_slot, raw_inputs[main_slot])
            if resume_data[main_slot] is None:
                resume_data[main_slot] = {}
            resume_data[main_slot].update(sub_params)

            # Check for missing sub-parameters
            missing_sub_params = [key for key in resume_template[main_slot].keys()
                                  if key not in resume_data[main_slot] or not resume_data[main_slot][key]]

            if missing_sub_params:
                # Generate follow-up question
                question = generate_followup_questions(main_slot, missing_sub_params)
                print(f"Bot: {question}")
                continue  # Continue collecting input
            else:
                # Main slot is completed
                main_slots_completed[main_slot] = True
                current_main_slot_index += 1
        else:
            # For list-type main slots, we've already handled them
            main_slots_completed[main_slot] = True
            current_main_slot_index += 1

    # After all main_slots are processed
    print("Bot: Thank you! All information has been collected.")

    # At the end, save the resume data to JSON
    with open('Resume_GPT/Multi-Turn-QA-System/output/resume.json', 'w') as f:
        json.dump(resume_data, f, indent=4)
    print("Bot: Your resume has been saved to 'resume.json'.")




if __name__ == "__main__":
    main()
