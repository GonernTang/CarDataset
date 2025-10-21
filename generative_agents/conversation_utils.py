import json, re, os
import random
import logging
from typing import List, Dict
from chat_api import run_chatgpt, run_chatgpt_with_examples
from prompts import *

CHARACTER_FROM_MSC_PROMPT = """
Let's create detailed driver profiles designed for conversations with an in-car AI assistant called CarBU-Agent.

Example persona format:
%s

Each driver persona should describe the individual’s lifestyle and daily driving habits in a way that helps CarBU-Agent understand how to personalize navigation, car settings, and conversation memory.

Include crucial biographical details:
- Name, age, gender, occupation, and marital status.
- Family and relationship context (e.g., spouse, children, coworkers, friends).
- Daily driving patterns (e.g., commute routes, usual driving times, preferred destinations).
- Behavioral traits while driving (e.g., cautious, adventurous, impatient, talkative).
- Interaction style with CarBU-Agent (e.g., friendly, efficient, relies heavily on memory recall).
- Preferences for in-car experiences (e.g., music genre, temperature, seat position, route types).
- Distinctive habits that may influence driving or memory (e.g., “often forgets to refuel,” “enjoys talking about travel plans,” “tracks mileage carefully”).

For the following attributes, write a detailed persona suitable for driver–CarBU-Agent interactions.

Output a valid JSON object with the following keys:
{
  "name": "<Driver’s full name>",
  "persona": "<A rich paragraph describing the driver’s profile and driving-related behavior>"
}

Attributes:
%s

Start your answer with a curly bracket.
"""


CASUAL_DIALOG_PROMPT = """ 
- Rewrite the following sentence to sound natural, polite, and human-like. 
- Keep it concise and conversational, as if spoken in a friendly driver–assistant dialogue. 
- Avoid slang or exaggerated casualness, but also avoid being too formal or stiff.
Input: %s\nOutput: 
"""

SESSION_SUMMARY_PROMPT = "Previous conversations between %s and car-agent so far can be summarized as follows: %s. The current time and date are %s. %s and car-agent just had the following conversation:\n\n%s\n\nSummarize the previous and current conversations between %s and car-agent in 150 words or less. Include key facts(time, location, action, people) about both speakers and time references.\n\n"


SESSION_SUMMARY_INIT_PROMPT = "Write a concise summary containing key facts(time, location, action, people) mentioned about %s and car-agent on %s in the following conversation:\n\n%s\n\n"


# VISUAL_QUESTION_PROMPT = "{}\n\n{}\n\n{} says, {}, and {}. Write the most natural question or comment {} can include in her response."


def get_driver_character(args):
    # check if personas exist, else generate persona + summary
    if os.path.exists(args.driver_file) and not args.overwrite_persona:
        return None
    else:
        # load all characters
        all_characters = json.load(open('./data/driver_characters_all.json', encoding='utf-8'))

        # pick a random index where in_dataset == 0
        candidates = [idx for idx, d in enumerate(all_characters['train']) if not d.get("in_dataset", 0)]
        if not candidates:
            raise RuntimeError("No available characters with in_dataset == 0")
        selected_idx = random.choice(candidates)

        attributes = all_characters['train'][selected_idx]

        # --- NEW: find the driver key dynamically (exclude 'in_dataset' and any metadata keys)
        # Prefer keys that start with 'Driver' (case-insensitive). Fallback to first non-in_dataset key.
        driver_keys = [k for k in attributes.keys() if k.lower().startswith('driver')]
        if not driver_keys:
            driver_keys = [k for k in attributes.keys() if k != "in_dataset"]
        if not driver_keys:
            raise RuntimeError(f"No driver key found in entry at index {selected_idx}: {attributes}")

        driver_key = driver_keys[0]  # choose the first matching key
        driver_prompt = attributes[driver_key]

        # mark as used and write back
        all_characters['train'][selected_idx]["in_dataset"] = 1
        with open('./data/driver_characters_all.json', "w", encoding='utf-8') as f:
            json.dump(all_characters, f, indent=2, ensure_ascii=False)

        # build agent using the dynamic driver prompt
        driver = get_character(args, driver_prompt)

        # Keep a summary and the original prompt lines
        driver['persona_summary'] = driver.get('persona', driver.get('name', driver_key))
        driver['msc_prompt'] = driver_prompt

        # delete persona if exists (original logic)
        if 'persona' in driver:
            del driver['persona']

        # set a name field if not present (optional): use explicit "My name is X." if available
        if 'name' not in driver:
            # try to extract name from the first line of the prompt if it starts with "My name is"
            if isinstance(driver_prompt, list) and len(driver_prompt) > 0 and isinstance(driver_prompt[0], str):
                first = driver_prompt[0].strip()
                if first.lower().startswith("my name is"):
                    # extract trailing text as name
                    agent_name = first[len("my name is"):].strip().strip(". ")
                    if agent_name:
                        driver['name'] = agent_name
                    else:
                        driver['name'] = driver_key
                else:
                    driver['name'] = driver_key
            else:
                driver['name'] = driver_key

        # LOGGING: brief info + detailed debug dump
        logging.info(
            "Generated Driver persona: driver_key=%s, name=%s, msc_prompt_preview=%s",
            driver_key,
            driver.get('name', '<unknown>'),
            (str(driver['msc_prompt'])[:80] + '...') if len(str(driver['msc_prompt'])) > 80 else str(driver['msc_prompt'])
        )
        try:
            driver_json = json.dumps(driver, indent=2, ensure_ascii=False)
        except Exception:
            driver_json = str(driver)
        logging.debug("Full Driver structure:\n%s", driver_json)

        print("Driver Persona: %s" % driver.get('persona_summary', driver.get('name', driver_key)))
    return driver

def get_character(args, attributes, target='human', ref_age=None):

    task = json.load(open(os.path.join(args.prompt_dir, 'persona_generation_examples.json')))
    persona_examples = [task["input_prefix"] + json.dumps(e["input"], indent=2) + '\n' + task["output_prefix"] + e["output"] for e in task['examples']]
    input_string = task["input_prefix"] + json.dumps(attributes, indent=2)

    query = CHARACTER_FROM_MSC_PROMPT % (persona_examples, input_string)

    try:
        output = run_chatgpt(query, num_gen=1, num_tokens_request=1000, use_16k=True).strip()
        output = json.loads(output)
    except:
        output = run_chatgpt(query, num_gen=1, num_tokens_request=1000, use_16k=True).strip()
        output = json.loads(output)
    
    if type(output) == list:
        output = [clean_json_output(out) for out in output]
    elif type(output) == str:
        output = clean_json_output(output)
    elif type(output) == dict:
        output = {k.lower(): v for k,v in output.items()}
        pass
    else:
        raise TypeError
    
    # print(output)

    return output

def get_CarBU_agent(CarAgent_path: str = "./data/CarAgent.json") -> Dict:
    """
    Load CarAgent.json and return a dict structured like driver, i.e. with:
      - name
      - persona_summary
      - msc_prompt (list[str])
      - (optional) persona removed if existed
    """
    with open(CarAgent_path, "r", encoding="utf-8") as f:
        CarAgent = json.load(f)

    # Ensure basic fields exist
    name = CarAgent.get("name", "CarBU-Agent")
    persona_summary = CarAgent.get("persona_summary", "")

    # Build msc_prompt from persona_summary: split into short, actionable lines.
    # You can customize this mapping to include explicit memory cues.
    msc_prompt: List[str] = [
        f"My name is {name}.",
        "I am an intelligent in-car assistant (AI, not human).",
        "I communicate in a calm, reliable, and context-aware manner.",
        "I recall user preferences, routines, and shared family contexts naturally.",
        "I avoid emotional speculation and focus on user comfort, efficiency, and safety.",
        "When referencing past events, I do so factually and politely."
    ]

    # Optionally, if you want to enrich msc_prompt from persona_summary by sentence-splitting:
    # sentences = [s.strip() for s in persona_summary.replace("\n", " ").split(".") if s.strip()]
    # msc_prompt = [("My name is " + name + ".")] + sentences

    # Construct final CarAgent structure matching driver style
    CarAgent_struct = {
        "name": name,
        "persona_summary": persona_summary,
        "msc_prompt": msc_prompt
    }

    # If there is 'persona' field and you want to follow the same logic as driver, remove it
    if "persona" in CarAgent_struct:
        del CarAgent_struct["persona"]

    return CarAgent_struct

def get_datetime_string(input_time='', input_date=''):

    assert input_time or input_date

    if input_date:
        year, month, day = input_date
    if input_time:
        hour, min = input_time
        time_mod = 'am' if hour <= 12 else 'pm'
        hour = hour if hour <= 12 else hour-12
        min = str(min).zfill(2)

    if input_time and not input_date:
        return str(hour) + ':' + min + ' ' + time_mod
    elif input_date and not input_time:
        return day + ' ' + month + ', ' + year
    else:
        return str(hour) + ':' + min + ' ' + time_mod + ' on ' + day + ' ' + month + ', ' + year 


def clean_dialog(output, name):

    if output.startswith(name):
        output = output[len(name):]
        output = output.strip()
        if output[0] == ':':
            output = output[1:]
            output = output.strip()
    
    return output


def clean_json_output(output_string):

    print(output_string)

    output_string = output_string.strip()

    if output_string[0] == '[' and output_string[-1] != ']':
        start_index = output_string.index('[')
        end_index = output_string.rindex(']')
        output_string = output_string[start_index:end_index+1]

    if output_string[0] == '{' and output_string[-1] != '}':
        start_index = output_string.index('{')
        end_index = output_string.rindex('}')
        output_string = output_string[start_index:end_index+1]

    # balance brackets in json
    num_start_bracket = len(find_indices(output_string, '{'))
    num_end_bracket = len(find_indices(output_string, '}'))

    if num_start_bracket != num_end_bracket:
        if num_end_bracket < num_start_bracket:
            output_string = output_string + ' '.join(['}']*(num_start_bracket-num_end_bracket))
        if num_start_bracket < num_end_bracket:
            output_string = ' '.join(['{']*(num_end_bracket-num_start_bracket)) + ' ' + output_string

    # balance brackets in json
    num_start_bracket = len(find_indices(output_string, '['))
    num_end_bracket = len(find_indices(output_string, ']'))

    if num_start_bracket != num_end_bracket:
        if num_end_bracket < num_start_bracket:
            output_string = output_string + ' '.join(['[']*(num_start_bracket-num_end_bracket))
        if num_start_bracket < num_end_bracket:
            output_string = ' '.join([']']*(num_end_bracket-num_start_bracket)) + ' ' + output_string

    return json.loads(output_string)


def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices

