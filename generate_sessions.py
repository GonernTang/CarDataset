import sys
import os
import json
import logging
import argparse
import random
from datetime import date, timedelta, datetime
from chat_api import set_openai_key, set_openai_base_url

from generative_agents.conversation_utils import *
from generative_agents.html_utils import convert_to_chat_html
from generative_agents.event_utils import *
from generative_agents.memory_utils import *
from prompts import *


os.environ["OPENAI_API_KEY"] = "sk-1Bd3OrhU3eg3S2wY5yKwerOlcpu4HuEXAerM8S7ybT4GVKJj"
os.environ["OPENAI_BASE_URL"] = "https://ai.nengyongai.cn/v1"
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"
os.environ["EMBEDDING_MODEL_NAME"] = "text-embedding-ada-002"


logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out-dir', required=True, type=str, help="Path to directory containing agent files and downloaded images for a conversation")
    parser.add_argument('--prompt-dir', required=True, type=str, help="Path to the dirctory containing in-context examples")
    
    parser.add_argument('--start-session', type=int, default=1, help="Start iterating from this index; first session is 1")
    parser.add_argument('--num-sessions', type=int, default=20, help="Maximum number of sessions in the conversation")
    parser.add_argument('--num-days', type=int, default=240, help="Desired temporal span of the multi-session conversation")
    parser.add_argument('--num-events', type=int, default=15, help="Total number of events to generate for each agent; 1 per session works best")
    parser.add_argument('--max-turns-per-session', type=int, default=8, help="Maximum number of total turns in each session")
    parser.add_argument('--num-events-per-session', type=int, default=2, help="Total number of events to be assigned to each agent per session; 1-2 works best")

    parser.add_argument('--persona', action="store_true", help="Set flag to sample a new persona from MSC and generate details")
    parser.add_argument('--session', action="store_true", help="Set flag to generate sessions based on the generated/existing personas")
    parser.add_argument('--events', action="store_true", help="Set flag to generate and events suited to the generated/existing personas")
    # parser.add_argument('--blip-caption', action="store_true", help="Set flag to use BLIP model to generate captions for downloaded images")
    parser.add_argument('--overwrite-persona', action='store_true', help="Overwrite existing persona summaries saved in the agent files")
    parser.add_argument('--overwrite-events', action='store_true', help="Overwrite existing events saved in the agent files")
    parser.add_argument('--overwrite-session', action='store_true', help="Overwrite existing sessions saved in the agent files")
    parser.add_argument('--summary', action="store_true", help="Set flag to generate and use summaries in the conversation generation prompt")

    parser.add_argument('--emb-file', type=str, default='embeddings.pkl', help="Name of the file used to save embeddings for the fine-grained retrieval-based memory module")
    parser.add_argument('--reflection', action="store_true", help="Set flag to use reflection module at the end of each session and include in the conversation generation prompt for context")

    args = parser.parse_args()
    return args



def save_agents(agents, args):

    driver, CarAgent = agents
    logging.info("Saving updated Driver to %s" % args.driver_file)
    with open(args.driver_file, 'w') as f:
        json.dump(driver, f, indent=2)
    logging.info("Saving updated CarAgent to %s" % args.CarAgent_file)
    with open(args.CarAgent_file, 'w') as f:
        json.dump(CarAgent, f, indent=2)


def load_agents(args):

    driver = json.load(open(args.driver_file))
    CarAgent = json.load(open(args.CarAgent_file))
    return driver, CarAgent



def get_random_time():

    start_time = timedelta(hours=9, minutes=0, seconds=0)
    end_time = timedelta(hours=21, minutes=59, seconds=59)
    random_seconds = random.randint(start_time.total_seconds(), end_time.total_seconds())
    hours = random_seconds//3600
    minutes = (random_seconds - (hours*3600))//60
    return timedelta(hours=hours, minutes=minutes, seconds=0)


def datetimeStr2Obj(dateStr):
    if 'am' in dateStr:
        datetimeObj = datetime.strptime(dateStr, "%H:%M am on %d %B, %Y")
    else:
        datetimeObj = datetime.strptime(dateStr, "%H:%M pm on %d %B, %Y")
    return datetimeObj

def datetimeObj2Str(datetimeObj):

    time_mod = 'am' if datetimeObj.hour <= 12 else 'pm'
    hour = datetimeObj.hour if datetimeObj.hour <= 12 else datetimeObj.hour-12
    min = str(datetimeObj.minute).zfill(2)
    return str(hour) + ':' + min + ' ' + time_mod + ' on ' + str(datetimeObj.day) + ' ' + datetimeObj.strftime("%B") + ', ' + str(datetimeObj.year)


def dateObj2Str(dateObj):
    return dateObj.strftime("%d") + ' ' + dateObj.strftime("%B") + ', ' + dateObj.strftime("%Y")


def get_random_date():

    # initializing dates ranges
    test_date1, test_date2 = date(2024, 1, 1), date(2024, 12, 30)
    # getting days between dates
    dates_bet = test_date2 - test_date1
    total_days = dates_bet.days
    delta_days = random.choice(range(1, total_days))
    random_date = test_date1 + timedelta(days=int(delta_days))
    return random_date



def get_session_summary(session, driver, curr_date, previous_summary=""):

    session_query = ''
    for c in session:
        session_query += "%s: %s\n" % (c["speaker"], c["text"])

    if previous_summary:

        query = SESSION_SUMMARY_PROMPT % (driver['name'], previous_summary, curr_date,
                                               driver['name'], session_query, driver['name'])
    else:
        query = SESSION_SUMMARY_INIT_PROMPT % (driver['name'], curr_date, session_query)

    query += '\n\n'
    # should summarize persona, previous conversations with respect to driver.
    output = run_chatgpt(query, 1, 150, 'chatgpt')
    output = output.strip()
    return output



def get_all_session_summary(speaker, curr_sess_id):

    summary = "\n"
    for sess_id in range(1, curr_sess_id):
        sess_date = speaker['session_%s_date_time' % sess_id]
        sess_date = sess_date[2] + ' ' + sess_date[1] + ', ' + sess_date[0]
        summary += sess_date + ': ' + speaker["session_%s_summary" % sess_id] + '\n'
    return summary



def catch_date(date_str):
    date_format1 = '%d %B, %Y'
    date_format2 = '%d %B %Y'
    try:
        return datetime.strptime(date_str, date_format1)
    except:
        return datetime.strptime(date_str, date_format2)


def get_session_date(events, args, prev_date = None):

    driver_events = events
    
    driver_events = sort_events_by_time(driver_events)
    curr_count = 0
    stop_count = args.num_events_per_session
    stop_date_a = None
    for e in driver_events:
        event_date =  catch_date(e['date'])
        if prev_date:
            if event_date >= prev_date:
                print("Including event %s for Driver" % json.dumps(e, indent=2))
                curr_count += 1
        else:
            print("Including event %s for Driver" % json.dumps(e, indent=2))
            curr_count += 1
        if curr_count == stop_count:
            stop_date_a = event_date
            break
    stop_date_a = event_date

    # return max(stop_date_a, stop_date_b) + timedelta(days=1)
    return stop_date_a + timedelta(days=random.choice([1, 2]))


def get_relevant_events(events, curr_date, prev_date=None):

    events = sort_events_by_time(events)
    relevant_events = []
    for e in events:
        # event_date = datetime.strptime(e['date'], "%d %B, %Y")
        event_date = catch_date(e['date'])
        if event_date > curr_date:
            continue
        if prev_date:
            if event_date <= prev_date:
                continue
        relevant_events.append(e)

    return relevant_events


def get_event_string(session_events, all_events):

    id2events = {e['id']: e for e in all_events}

    event_string = ""
    for e in session_events:
        try:
            event_text = 'On' + e["date"] + ", " + e["sub-event"]
        except KeyError:
            event_text = 'On' + e["date"] + ", " + e["sub_event"]

        # if the event is caused by previous events, include them for context
        if len(e['caused_by']) > 0:
            event_text += ' Because previously'
            for e_id in e['caused_by']:
                try:
                    event_text += ', ' + id2events[e_id]["sub-event"] + ' (%s)' % id2events[e_id]["date"]
                except KeyError:
                    event_text += ', ' + id2events[e_id]["sub_event"] + ' (%s)' % id2events[e_id]["date"]
        
        event_string += event_text + "\n"

    return event_string


def remove_context(args, curr_dialog, prev_dialog, caption=None):

    prompt_data = json.load(open(os.path.join(args.prompt_dir, 'remove_context_examples.json')))

    query = prompt_data["input_format"].format(prev_dialog, curr_dialog)
    output = run_chatgpt_with_examples(prompt_data["prompt"], 
                              [[prompt_data["input_format"].format(*example["input"]) if len(example["input"]) == 2 else prompt_data["input_format_w_image"].format(*example["input"]), example["output"]] for example in prompt_data['examples']], 
                              query, num_gen=1, num_tokens_request=128, use_16k=False)
    return output


def get_driver_speak(driver, curr_sess_id=0, 
                    prev_sess_date_time='', curr_sess_date_time='', 
                    use_events=False, instruct_stop=False, dialog_id=0, last_dialog='', embeddings=None, reflection=False):
    
    stop_instruction = "To end the conversation, write [END] at the end of the dialog."
    if instruct_stop:
        print("**** Using stop instruction ****")

    if curr_sess_id == 1:
        #第一次对话
        if use_events:
            #如果driver要引用过去自己做过的事的话，他应该向CarAgent询问过去的事，让CarAgent回忆并告诉driver具体的细节
            events = get_event_string(driver['events_session_%s' % curr_sess_id], driver['graph'])
            speak = DRIVER_CONV_PROMPT_SESS_1_EVENTS % (driver['persona_summary'], 
                                                        driver['name'], 
                                                        curr_sess_date_time,
                                                        events,
                                                        driver['name'],
                                                        stop_instruction if instruct_stop else '')
        else:
            #不用的话，driver就向CarAgent咨询日常驾驶情景中会发生的情况，产生新的events
            speak = DRIVER_CONV_PROMPT_SESS_1 % (driver['persona_summary'],
                                                 driver['name'], 
                                                 curr_sess_date_time,
                                                 stop_instruction if instruct_stop else '')
    else:
        #不是第一次对话
        if use_events:
            events = get_event_string(driver['events_session_%s' % curr_sess_id], driver['graph'])
            if dialog_id == 0:
                context_from_1 = get_recent_context(driver, curr_sess_id, reflection=reflection)
                recent_context = '\n'.join(context_from_1) # with reflection
                speak = DRIVER_CONV_PROMPT_EVENTS_INIT % (driver['persona_summary'],
                                                   driver['name'],  
                                                   prev_sess_date_time,
                                                   curr_sess_date_time, 
                                                   driver['name'],  
                                                   driver['session_%s_summary' % (curr_sess_id-1)], 
                                                   driver['name'],
                                                   events, 
                                                   stop_instruction if instruct_stop else '')
            else:
                past_context = get_relevant_context(driver, last_dialog, embeddings, curr_sess_id, reflection=reflection)
                speak = DRIVER_CONV_PROMPT_EVENTS % (driver['persona_summary'],
                                                    driver['name'],  
                                                    prev_sess_date_time,
                                                    curr_sess_date_time, 
                                                    driver['name'],  
                                                    driver['session_%s_summary' % (curr_sess_id-1)], 
                                                    driver['name'],
                                                    events, 
                                                    past_context,
                                                    stop_instruction if instruct_stop else '')
        else:
            summary = get_all_session_summary(driver, curr_sess_id)
            speak = DRIVER_CONV_PROMPT % (driver['persona_summary'],
                                         driver['name'], 
                                         prev_sess_date_time, 
                                         curr_sess_date_time, 
                                         driver['name'],
                                         summary,  
                                         driver['name'],
                                         stop_instruction if instruct_stop else '')

    return speak


def get_CarAgent_speak(driver, curr_sess_id=0, 
                    prev_sess_date_time='', curr_sess_date_time='', 
                    use_events=False, instruct_stop=False, dialog_id=0, last_dialog='', embeddings=None, reflection=False):
    
    stop_instruction = "To end the conversation, write [END] at the end of the dialog."
    if instruct_stop:
        print("**** Using stop instruction ****")
    

    if curr_sess_id == 1:
        speak = CARAGENT_CONV_PROMPT_SESS_1 % (driver['name'],
                                                curr_sess_date_time,
                                                stop_instruction if instruct_stop else '')
    else:
        events = get_event_string(driver['events_session_%s' % curr_sess_id], driver['graph'])
        summary = get_all_session_summary(driver, curr_sess_id)
        past_context = get_relevant_context(driver, last_dialog, embeddings, curr_sess_id, reflection=reflection)
        speak = CARAGENT_CONV_PROMPT % (driver['name'],
                                        curr_sess_date_time,
                                        driver['name'],
                                        summary,
                                        driver['name'],
                                        events,
                                        past_context,
                                        stop_instruction if instruct_stop else '')

    return speak


def get_session(driver, CarAgent, args, prev_date_time_string='', curr_date_time_string='', curr_sess_id=0, reflection=False):
    
    # load embeddings for retrieveing relevat observations from previous conversations
    if curr_sess_id == 1:
        embeddings = None
    else:
        embeddings = pkl.load(open(args.emb_file, 'rb'))

    curr_speaker = 0
    conv_so_far = driver['name'] + ': '
    session = []

    # choose a random turn number to include instructions for ending the session
    stop_dialog_count = args.max_turns_per_session if args.max_turns_per_session <= 10 else random.choice(list(range(10, args.max_turns_per_session))) # choose a random turn number to include instructions for ending the session
    break_at_next_a = False
    break_at_next_b = False

    for i in range(args.max_turns_per_session):

        if break_at_next_a and break_at_next_b:
            break
        
        # if curr_speaker == 0:
        #     agent_query = get_agent_query(driver, CarAgent, prev_sess_date_time=prev_date_time_string, curr_sess_date_time=curr_date_time_string,
        #                             curr_sess_id=curr_sess_id, use_events=args.events, instruct_stop=i>=stop_dialog_count, 
        #                             dialog_id=i, last_dialog='' if i == 0 else session[-1]['speaker'] + ' says, ' + session[-1]['clean_text'], 
        #                             embeddings=embeddings, reflection=reflection)
        
        if curr_speaker ==0:
            agent_query = get_driver_speak(driver, curr_sess_id=curr_sess_id, 
                    prev_sess_date_time=prev_date_time_string, curr_sess_date_time=curr_date_time_string, 
                    use_events=args.events, instruct_stop=i>stop_dialog_count, dialog_id=i, 
                    last_dialog='' if i == 0 else session[-1]['speaker'] + ' says ' + session[-1]['text'], 
                    embeddings=embeddings, reflection=reflection)
        else:
            agent_query = get_CarAgent_speak(driver, curr_sess_id=curr_sess_id, 
                    prev_sess_date_time=prev_date_time_string, curr_sess_date_time=curr_date_time_string, 
                    use_events=args.events, instruct_stop=i>stop_dialog_count, dialog_id=i, 
                    last_dialog='' if i == 0 else session[-1]['speaker'] + ' says ' + session[-1]['text'], 
                    embeddings=embeddings, reflection=reflection)

        output = run_chatgpt(agent_query + conv_so_far, 1, 100, 'chatgpt', temperature=1.2)
        output = output.strip().split('\n')[0]
        output = clean_dialog(output, driver['name'] if curr_speaker == 0 else CarAgent['name'])
        output = {
            "text": output, 
            # "raw_text": output,
            "speaker": driver['name'] if curr_speaker == 0 else CarAgent['name'],
            "dia_id": 'D%s:%s' % (curr_sess_id, i+1)
            }
        # output["speaker"] = driver["name"] if curr_speaker == 0 else CarAgent['name']
    
        # output["dia_id"] = 'D%s:%s' % (curr_sess_id, i+1)
        session.append(output)

        # print(output)

        # conv_so_far = conv_so_far + output["text"] + '\n'
        conv_so_far = conv_so_far + output["text"] + '\n'



        if output['text'].endswith('[END]'):
            if curr_speaker == 0:
                break_at_next_a = True
            else:
                break_at_next_b = True

        conv_so_far += f"\n{CarAgent['name']}: " if curr_speaker == 0 else f"\n{driver['name']}: "
        curr_speaker = int(not curr_speaker)

    return session




def main():

    # get arguments
    args = parse_args()

    set_openai_key()
    set_openai_base_url()

    args.emb_file = os.path.join(args.out_dir, args.emb_file)

    # create dataset directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    logging.info("Dataset directory: %s" % args.out_dir)

    args.driver_file = os.path.join(args.out_dir, 'driver.json')
    args.CarAgent_file = os.path.join(args.out_dir, 'CarAgent.json')

    
    # Step 1: Get personalities for the driver; get a randomly selected sample from the MSC dataset and expand the few-liner personas into detailed personas.
    if args.persona:
        driver = get_driver_character(args)

    CarAgent = get_CarBU_agent()
        
    if driver is not None and CarAgent is not None:
        save_agents([driver, CarAgent], args)


    # Step 2: check if events exist; if not, generate event graphs for each of the agents 
    if args.events:

        driver, CarAgent = load_agents(args)

        if 'graph' in driver and not args.overwrite_events:
            pass
        else:
            # if 'session_1_date_time' not in driver:
            start_date = get_random_date() # select a random date in 2022-2023
            end_date = start_date + timedelta(days=args.num_days)
            start_date = dateObj2Str(start_date)
            end_date = dateObj2Str(end_date)
            driver['events_start_date'] = start_date
            logging.info("Generating a random start date for the conversation")
            save_agents([driver, CarAgent], args)

            
            driver_events = []

            logging.info("Generating events for driver")
            trials = 0
            while len(driver_events) < args.num_events:
                logging.info("(Re)trying to generate events with dense causal connections: trial %s" % trials)
                driver_events = get_events(driver, start_date, end_date, args)
                driver["graph"] = driver_events
                trials += 1

            save_agents([driver, CarAgent], args)

        # make sure keys are all lower case
        driver_events = driver['graph']
        driver_events = [{k.lower(): v for k,v in e.items()} for e in driver_events]
        driver["graph"] = driver_events

        save_agents([driver, CarAgent], args)

    # Step 3: 
    if args.session:

        driver, CarAgent = load_agents(args)

        # default start index is 1; if resuming conversation from a leter session, indicate in script arguments using --start-session
        for j in range(args.start_session, args.num_sessions+1):

            print("******************* SESSION %s ******************" % j)

            if 'session_%s' % j not in driver or args.overwrite_session:

                if j>1:
                    prev_date_time = datetimeStr2Obj(driver['session_%s_date_time' % (j-1)])
                    prev_date_time_string = driver['session_%s_date_time' % (j-1)]
                else:
                    prev_date_time, prev_date_time_string = None, None

                # get conversation date and time for each session
                curr_time = get_random_time() # timedelta object
                curr_date = get_session_date(driver['graph'], args, prev_date=prev_date_time) # datetime object
                curr_date_time = curr_date + curr_time # datetime object
                
                relevant_events_driver = get_relevant_events(driver['graph'],  curr_date_time, prev_date=prev_date_time)
                driver['events_session_%s' % j] = relevant_events_driver

                if len(relevant_events_driver) == 0:
                    logging.info("Stoppping conversation because no more events available in KG.")
                    break

                curr_date_time_string = datetimeObj2Str(curr_date_time)
                driver['session_%s_date_time' % j] = curr_date_time_string
                CarAgent['session_%s_date_time' % j] = curr_date_time_string
                save_agents([driver, CarAgent], args)
                
                session = get_session(driver, CarAgent, args,
                                      prev_date_time_string=prev_date_time_string, curr_date_time_string=curr_date_time_string, 
                                      curr_sess_id=j, reflection=args.reflection)
                
                driver['session_%s' % j] = session
                CarAgent['session_%s' % j] = session

                save_agents([driver, CarAgent], args)

            if 'session_%s_facts' % j not in driver or args.overwrite_session:

                facts = get_session_facts(args, driver, j)

                driver['session_%s_facts' % j] = facts

                print(" --------- Session %s Summary for Driver---------" % (j))
                print(facts)

                save_agents([driver, CarAgent], args)

            if args.reflection and ('session_%s_reflection' % j not in driver or args.overwrite_session):

                reflections = get_session_reflection(args, driver, CarAgent, j)

                driver['session_%s_reflection' % j] = reflections['a']
                CarAgent['session_%s_reflection' % j] = reflections['b']

                print(" --------- Session %s Reflection for Agent A---------" % (j))
                print(reflections)

                save_agents([driver, CarAgent], args)

            if args.summary and ('session_%s_summary' % j not in driver or args.overwrite_session):

                summary = get_session_summary(driver['session_%s' % j], driver, driver['session_%s_date_time' % j], 
                                              previous_summary=None if j==1 else driver['session_%s_summary' % (j-1)])

                driver['session_%s_summary' % j] = summary

                save_agents([driver, CarAgent], args)

    driver, CarAgent = load_agents(args)
    convert_to_chat_html(driver, CarAgent, outfile=os.path.join(args.out_dir, 'sessions.html'), use_events=args.events, img_dir=args.out_dir)


if __name__ == "__main__":
    main()