import os, json
import time
import openai
import logging
from datetime import datetime
from chat_api import run_json_trials
import numpy as np
import pickle as pkl
import random
logging.basicConfig(level=logging.INFO)


REFLECTION_INIT_PROMPT = "{}\n\nGiven the information above, what are the three most salient insights that {} has about {}? Give concise answers in the form of a json list where each entry is a string."

REFLECTION_CONTINUE_PROMPT = "{} has the following insights about {} from previous interactions.{}\n\nTheir next conversation is as follows:\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about {} now? Give concise answers in the form of a json list where each entry is a string."

SELF_REFLECTION_INIT_PROMPT = "{}\n\nGiven the information above, what are the three most salient insights that {} has about self? Give concise answers in the form of a json list where each entry is a string."

SELF_REFLECTION_CONTINUE_PROMPT = "{} has the following insights about self.{}\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about self now? Give concise answers in the form of a json list where each entry is a string."


CONVERSATION2FACTS_PROMPT = """
Write a concise and short list of all possible OBSERVATIONS about each speaker that can be gathered from the CONVERSATION. Each dialog in the conversation contains a dialogue id within square brackets. Each observation should contain a piece of information about the speaker, and also include the dialog id of the dialogs from which the information is taken. The OBSERVATIONS should be objective factual information about the speaker that can be used as a database about them. Avoid abstract observations about the dynamics between the two speakers such as 'speaker is supportive', 'speaker appreciates' etc. Do not leave out any information from the CONVERSATION. Important: Escape all double-quote characters within string output with backslash.\n\n
"""


RETRIEVAL_MODEL = "text-embedding-ada-002" # contriever dragon dpr


def get_embedding(texts, model="text-embedding-ada-002"):
   texts = [text.replace("\n", " ") for text in texts]
   return np.array([openai.Embedding.create(input = texts, model=model)['data'][i]['embedding'] for i in range(len(texts))])


def get_session_facts(args, agent_a, agent_b, session_idx, return_embeddings=True):

    # Step 1: get events
    task = json.load(open(os.path.join(args.prompt_dir, 'fact_generation_examples_new.json')))
    query = CONVERSATION2FACTS_PROMPT
    examples = [[task['input_prefix'] + e["input"], json.dumps(e["output"], indent=2)] for e in task['examples']]

    conversation = ""
    conversation += agent_a['session_%s_date_time' % session_idx] + '\n'
    for i, dialog in enumerate(agent_a['session_%s' % session_idx]):
        try:
            conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"'
        except KeyError:
            conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog['text'] + '\"'

        conversation += '\n'
    
    # print(conversation)
    
    input = task['input_prefix'] + conversation
    facts = run_json_trials(query, num_gen=1, num_tokens_request=500, use_16k=False, examples=examples, input=input)

    if not return_embeddings:
        return facts

    agent_a_embeddings = get_embedding([agent_a['session_%s_date_time' % session_idx] + ', ' + f for f, _ in facts[agent_a['name']]])
    # agent_b_embeddings = get_embedding([agent_b['session_%s_date_time' % session_idx] + ', ' + f for f, _ in facts[agent_b['name']]])

    if session_idx > 1:
        with open(args.emb_file, 'rb') as f:
            embs = pkl.load(f)
    
        embs[agent_a['name']] = np.concatenate([embs[agent_a['name']], agent_a_embeddings], axis=0)
        # embs[agent_b['name']] = np.concatenate([embs[agent_b['name']], agent_b_embeddings], axis=0)
    else:
        embs = {}
        embs[agent_a['name']] = agent_a_embeddings
        # embs[agent_b['name']] = agent_b_embeddings
    
    with open(args.emb_file, 'wb') as f:
        pkl.dump(embs, f)
    
    return facts


def get_session_reflection(args, agent_a, agent_b, session_idx):


    # Step 1: get conversation
    conversation = ""
    conversation += agent_a['session_%s_date_time' % session_idx] + '\n'
    for dialog in agent_a['session_%s' % session_idx]:
        # if 'clean_text' in dialog:
        #     writer.write(dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"\n')
        # else:
        conversation += dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"\n'


    # Step 2: Self-reflections
    if session_idx == 1:
        agent_a_self = run_json_trials(SELF_REFLECTION_INIT_PROMPT.format(conversation, agent_a['name']), model='chatgpt', num_tokens_request=300)
        agent_b_self = run_json_trials(SELF_REFLECTION_INIT_PROMPT.format(conversation, agent_b['name']), model='chatgpt', num_tokens_request=300)

    else:
        agent_a_self = run_json_trials(SELF_REFLECTION_CONTINUE_PROMPT.format(agent_a['name'], '\n'.join(agent_a['session_%s_reflection' % (session_idx-1)]['self']), conversation, agent_a['name']), model='chatgpt', num_tokens_request=300)
        agent_b_self = run_json_trials(SELF_REFLECTION_CONTINUE_PROMPT.format(agent_b['name'], '\n'.join(agent_b['session_%s_reflection' % (session_idx-1)]['self']), conversation, agent_b['name']), model='chatgpt', num_tokens_request=300)

    # Step 3: Reflection about other speaker
    if session_idx == 1:
        agent_a_on_b = run_json_trials(REFLECTION_INIT_PROMPT.format(conversation, agent_a['name'], agent_b['name']), model='chatgpt', num_tokens_request=300)
        agent_b_on_a = run_json_trials(REFLECTION_INIT_PROMPT.format(conversation, agent_b['name'], agent_a['name']), model='chatgpt', num_tokens_request=300)

    else:
        agent_a_on_b = run_json_trials(REFLECTION_CONTINUE_PROMPT.format(agent_a['name'], agent_b['name'], '\n'.join(agent_a['session_%s_reflection' % (session_idx-1)]['other']), conversation, agent_a['name'], agent_b['name']), model='chatgpt', num_tokens_request=300)
        agent_b_on_a = run_json_trials(REFLECTION_CONTINUE_PROMPT.format(agent_b['name'], agent_a['name'], '\n'.join(agent_b['session_%s_reflection' % (session_idx-1)]['other']), conversation, agent_b['name'], agent_a['name']), model='chatgpt', num_tokens_request=300)

    if type(agent_a_self) == dict:
        agent_a_self = list(agent_a_self.values())
    if type(agent_b_self) == dict:
        agent_b_self = list(agent_b_self.values())
    if type(agent_a_on_b) == dict:
        agent_a_on_b = list(agent_a_on_b.values())
    if type(agent_b_on_a) == dict:
        agent_b_on_a = list(agent_b_on_a.values())  

    reflections = {}
    reflections['a'] = {'self': agent_a_self, 'other': agent_a_on_b}
    reflections['b'] = {'self': agent_b_self, 'other': agent_b_on_a}

    return reflections


def get_recent_context(driver, sess_id, context_length=2, reflection=False):

    speaker_1_facts = []
    for i in range(1, sess_id):
        speaker_1_facts += [driver['session_%s_date_time' % i] + ': ' + f for f, _ in driver['session_%s_facts' % i][driver["name"]]]

    if reflection:
        print(speaker_1_facts[-context_length:])
        print(driver['session_%s_reflection' % (sess_id-1)]['self'])
        return speaker_1_facts[-context_length:] + driver['session_%s_reflection' % (sess_id-1)]['self']
    else:
        return speaker_1_facts[-context_length:]


def get_relevant_context(driver, input_dialogue, embeddings, sess_id, context_length=2, reflection=False):

    logging.info("Getting relevant context for response to %s (session %s)" % (input_dialogue, sess_id))
    contexts_a = get_recent_context(driver, sess_id, 10000)
    # embeddings = pkl.load(open(emb_file, 'rb'))
    input_embedding = get_embedding([input_dialogue])
    sims_with_context_a = np.dot(embeddings[driver['name']], input_embedding[0])
    top_k_sims_a = np.argsort(sims_with_context_a)[::-1][:context_length]
    # print(sims_with_context_a, sims_with_context_b)
    if reflection:
        print([contexts_a[idx] for idx in top_k_sims_a])
        print( driver['session_%s_reflection' % (sess_id-1)]['self'])
        return [contexts_a[idx] for idx in top_k_sims_a] + random.sample(driver['session_%s_reflection' % (sess_id-1)]['self'], k=context_length//2)
    else:
        return [contexts_a[idx] for idx in top_k_sims_a]

