import os, json
import time
import openai
import logging
from datetime import datetime
from chat_api import run_json_trials
import numpy as np
import pickle as pkl
import random
import pprint

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


def get_session_facts(args, agent_a, session_idx, return_embeddings=True):
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

    input = task['input_prefix'] + conversation
    facts = run_json_trials(query, num_gen=1, num_tokens_request=500, use_16k=False, examples=examples, input=input)

    if not return_embeddings:
        return facts

    # --- SAFE: extract fact strings for this agent ---
    agent_name = agent_a['name']
    agent_facts = facts.get(agent_name, [])  # expect list of (fact_str, metadata) pairs or similar
    # If no facts found, return facts and skip embedding update
    if not agent_facts:
        logging.info("No facts generated for %s in session %s. Skipping embedding update.", agent_name, session_idx)
        return facts

    # create text list to embed: "date, fact"
    texts_to_embed = [agent_a['session_%s_date_time' % session_idx] + ', ' + f for f, _ in agent_facts]

    # get embeddings (wrap in try to catch failures)
    agent_a_embeddings = get_embedding(texts_to_embed)
    if agent_a_embeddings is None:
        logging.warning("get_embedding returned None for %s session %s; skipping embedding persistence.", agent_name, session_idx)
        return facts

    # ensure numpy array and 2D shape (n, dim)
    agent_a_embeddings = np.asarray(agent_a_embeddings)
    if agent_a_embeddings.ndim == 1:
        # single vector -> reshape to (1, dim)
        agent_a_embeddings = agent_a_embeddings.reshape(1, -1)
    elif agent_a_embeddings.ndim > 2:
        # unexpected shape: try to flatten leading dims to 2D
        agent_a_embeddings = agent_a_embeddings.reshape(agent_a_embeddings.shape[0], -1)

    # load or create embeddings store
    if session_idx > 1 and os.path.exists(args.emb_file):
        with open(args.emb_file, 'rb') as f:
            embs = pkl.load(f)
        logging.info("Loaded existing embeddings from %s", args.emb_file)
    else:
        embs = {}

    # if agent already has embeddings, ensure compatibility in dimension
    if agent_name in embs and embs[agent_name] is not None:
        existing = np.asarray(embs[agent_name])
        # existing should be 2D
        if existing.ndim == 1:
            existing = existing.reshape(1, -1)

        # shape check: dims must match
        if existing.shape[1] != agent_a_embeddings.shape[1]:
            logging.warning("Embedding dimension mismatch for %s: existing dim=%s new dim=%s. Attempting to handle by reprojecting or skipping.",
                            agent_name, existing.shape[1], agent_a_embeddings.shape[1])
            # Simple safe fallback: skip concatenation if dims don't match
            # (Better: re-generate with consistent embedding model or handle projection)
            embs[agent_name] = agent_a_embeddings  # replace to keep latest consistent set
        else:
            embs[agent_name] = np.concatenate([existing, agent_a_embeddings], axis=0)
    else:
        # no existing embeddings for this agent
        embs[agent_name] = agent_a_embeddings

    # persist
    with open(args.emb_file, 'wb') as f:
        pkl.dump(embs, f)

    # debug print
    logging.info("Saved embeddings for %s. New shape: %s", agent_name, embs[agent_name].shape)

    # optionally pretty-print facts for debugging
    pprint.pprint(facts.get(agent_a['name']))
    return facts

# def get_session_facts(args, agent_a, session_idx, return_embeddings=True):

#     # Step 1: get events
#     task = json.load(open(os.path.join(args.prompt_dir, 'fact_generation_examples_new.json')))
#     query = CONVERSATION2FACTS_PROMPT
#     examples = [[task['input_prefix'] + e["input"], json.dumps(e["output"], indent=2)] for e in task['examples']]

#     conversation = ""
#     conversation += agent_a['session_%s_date_time' % session_idx] + '\n'
#     for i, dialog in enumerate(agent_a['session_%s' % session_idx]):
#         try:
#             conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"'
#         except KeyError:
#             conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog['text'] + '\"'

#         conversation += '\n'
    
#     # print(conversation)
    
#     input = task['input_prefix'] + conversation
#     facts = run_json_trials(query, num_gen=1, num_tokens_request=500, use_16k=False, examples=examples, input=input)

#     if not return_embeddings:
#         return facts

#     pprint.pprint(facts.get(agent_a['name']))
#     agent_a_embeddings = get_embedding([agent_a['session_%s_date_time' % session_idx] + ', ' + f for f, _ in facts[agent_a['name']]])
#     # agent_b_embeddings = get_embedding([agent_b['session_%s_date_time' % session_idx] + ', ' + f for f, _ in facts[agent_b['name']]])

#     if session_idx > 1:
#         with open(args.emb_file, 'rb') as f:
#             embs = pkl.load(f)

#         logging.info("Loaded existing embeddings from %s" % agent_a_embeddings)
#         logging.info("Existing embeddings for %s has shape %s" % (agent_a['name'], embs[agent_a['name']].shape))
#         embs[agent_a['name']] = np.concatenate([embs[agent_a['name']], agent_a_embeddings], axis=0)
#         # embs[agent_b['name']] = np.concatenate([embs[agent_b['name']], agent_b_embeddings], axis=0)
#     else:
#         embs = {}
#         embs[agent_a['name']] = agent_a_embeddings
#         # embs[agent_b['name']] = agent_b_embeddings
    
#     with open(args.emb_file, 'wb') as f:
#         pkl.dump(embs, f)
    
#     return facts


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

