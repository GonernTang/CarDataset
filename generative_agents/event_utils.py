import os, json
import re
import ast
from typing import Optional, Tuple
import time
import openai
import logging
from datetime import datetime
from chat_api import run_chatgpt
import tiktoken
logging.basicConfig(level=logging.INFO)



EVENT_KG_FROM_PERSONA_PROMPT_SEQUENTIAL_INIT = """
Let's write a graph representing sub-events that occur in a person's driving experiences based on a short summary of their personality and driving habits. Nodes represent sub-events and edges represent the influence of past sub-events on a current sub-event.
- The graph is represented in the form of a json list. 
- Each entry is a dictionary containing the following keys: "sub-event", "time", "caused_by", "id". 
- The "sub-event" field contains a short description of the sub-event. 
- The "time" field contains a date.
- The "id" field contains a unique identifier for the sub-event.
- The "caused_by" field represents edges and is a list of "id" of existing sub-events that have caused this sub-event. Sub-events in the "caused_by" field should occur on dates before the sub-event they have caused. Generate as many causal connections as possible.
- An example of a causal effect is when the sub-event "started a vegetable garden" causes "harvested tomatoes".
- Sub-events can be positive or negative life events.

For example,

PERSONALITY: %s
OUTPUT: %s

----------------------------------------------------------------------------------------------------------------

For the following input personality and driving habits, generate three independent sub-events E1, E2 and E3 aligned with their personality and driving habits. Sub-events can be events under different driving scenarios and should reflect evolution in the person's relationships, state of mind, personality and driving habits etc. 

PERSONALITY: %s
OUTPUT: 
"""



EVENT_KG_FROM_PERSONA_PROMPT_SEQUENTIAL_CONTINUE = """
Let's write a graph representing sub-events that occur in a person's driving experiences based on a short summary of their personality and driving habits. Nodes represent sub-events and edges represent the influence of past sub-events on a current sub-event.
- The graph is represented in the form of a json list. 
- Each entry is a dictionary containing the following keys: "sub-event", "time", "caused_by", "id". 
- The "sub-event" field contains a short description of the sub-event. 
- The "time" field contains a date.
- The "id" field contains a unique identifier for the sub-event.
- The "caused_by" field represents edges and is a list of "id" of existing sub-events that have caused this sub-event. Sub-events in the "caused_by" field should occur on dates before the sub-event they have caused. Generate as many causal connections as possible.
- An example of a causal effect is when the sub-event "started a vegetable garden" causes "harvested tomatoes".
- Sub-events can be positive or negative life events.


For example,

PERSONALITY: %s
OUTPUT: %s

----------------------------------------------------------------------------------------------------------------

For the following input personality and driving habits,generate new sub-events %s that are caused by one or more EXISTING sub-events. Sub-events can be events under different driving scenarios and should reflect evolution in the person's relationships, state of mind, personality and driving habits etc. Do not repeat existing sub-events. Start and end your answer with a square bracket.

PERSONALITY: %s
EXISTING: %s
OUTPUT:  
"""

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding_name = 'cl100k_base' if model_name in ['gpt-4', 'gpt-3.5-turbo', 'text-embedding-ada-002'] else 'p50k_base'
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def sort_events_by_time(graph):

    def catch_date(date_str):
        date_format1 = '%d %B, %Y'
        date_format2 = '%d %B %Y'
        try:
            return datetime.strptime(date_str, date_format1)
        except:
            return datetime.strptime(date_str, date_format2)
    
    dates = [catch_date(node['date']) for node in graph]
    sorted_dates = sorted(enumerate(dates), key=lambda t: t[1])
    graph = [graph[idx] for idx, _ in sorted_dates]
    return graph




def _find_bracketed_substrings(s: str) -> list:
    """返回字符串中可能的 JSON-like 大括号/方括号子串候选（按发现顺序）"""
    candidates = []
    # 尝试寻找最外层的 [...] 或 {...}
    # 找第一个 '[' 或 '{'，并找到与之匹配的最后一个 ']' 或 '}'
    first_idx = None
    for i, ch in enumerate(s):
        if ch in ['[', '{']:
            first_idx = i
            opening = ch
            break
    if first_idx is None:
        return candidates
    # 根据 opening 找最后一个匹配的闭合符
    closing = ']' if opening == '[' else '}'
    last_idx = s.rfind(closing)
    if last_idx != -1 and last_idx > first_idx:
        candidates.append(s[first_idx:last_idx+1])

    # 备选：尝试用正则找所有 { ... } 或 [ ... ] 的完整段落
    for match in re.finditer(r'(\{[\s\S]*?\}|\[[\s\S]*?\])', s):
        candidates.append(match.group(0))
    # 去重并保留顺序
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq

def safe_parse_json_from_string(s: str) -> Tuple[Optional[object], str]:
    """
    尝试从字符串 s 中解析 JSON，返回 (parsed_obj, used_candidate_str).
    解析策略（按顺序）：
      1. 直接 json.loads(s)
      2. 从 s 中提取可能的 {...} 或 [...] 子串尝试 json.loads
      3. 对候选子串替换单引号为双引号并尝试 json.loads
      4. 使用 ast.literal_eval 作为后备（能处理 True/False/None 和单引号）
    若都失败，返回 (None, last_candidate_tried_or_original_string)
    同时会捕获并不抛出异常（上层可根据返回值决定重试/报错）。
    """
    if not isinstance(s, str):
        return None, ""

    s_strip = s.strip()
    if s_strip == "":
        return None, s

    # 1) 直接尝试
    try:
        return json.loads(s_strip), s_strip
    except Exception:
        pass

    # 2) 提取候选
    candidates = _find_bracketed_substrings(s_strip)
    # 若没匹配到任何候选，将原字符串放入候选以便后续处理尝试
    if not candidates:
        candidates = [s_strip]

    last_candidate = candidates[-1]
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            return parsed, cand
        except Exception:
            # 3) 单引号 -> 双引号 的简单替换后重试（谨慎）
            cand2 = cand.replace("'", '"')
            # 修复常见的 None/True/False（JSON 需要 null/true/false）
            cand2 = re.sub(r'\bNone\b', 'null', cand2)
            cand2 = re.sub(r'\bTrue\b', 'true', cand2)
            cand2 = re.sub(r'\bFalse\b', 'false', cand2)
            try:
                parsed = json.loads(cand2)
                return parsed, cand2
            except Exception:
                pass
            # 4) ast.literal_eval 作为最后手段（能解析 Python 字面量）
            try:
                parsed = ast.literal_eval(cand)
                return parsed, cand
            except Exception:
                pass
            last_candidate = cand

    # 全部失败
    return None, last_candidate


# get events in one initialization step and one or more continuation steps.
def get_events(agent, start_date, end_date, args):


    task = json.load(open(os.path.join(args.prompt_dir, 'event_generation_examples.json')))
    persona_examples = [e["input"] + '\nGenerate events between 1 January, 2024 and 30 December, 2025.' for e in task['examples']]
    
    # Step 1: Get initial events
    task = json.load(open(os.path.join(args.prompt_dir, 'graph_generation_examples.json')))
    input = agent['persona_summary'] + '\nAssign dates between %s and %s.' % (start_date, end_date)
    query = EVENT_KG_FROM_PERSONA_PROMPT_SEQUENTIAL_INIT % (persona_examples[0], 
                                                                   json.dumps(task['examples'][0]["output"][:12], indent=2),
                                                                   input)
    logging.info("Generating initial events")
    try:
        output = run_chatgpt(query, num_gen=1, num_tokens_request=512, use_16k=False, temperature=1.0).strip()
        output = json.loads(output)
    except:
        output = run_chatgpt(query, num_gen=1, num_tokens_request=512, use_16k=False, temperature=1.0).strip()
        # output = json.loads(output)
        parsed, used_candidate = safe_parse_json_from_string(output)
        if parsed is None:
            # 打印尽可能多的调试信息，便于定位 LLM 输出不规范的原因
            print("ERROR: Failed to parse JSON from model output in get_events.")
            print("Raw model output:")
            print(output)
            print("Tried candidate (last):")
            print(used_candidate)
            # 你可以选择：1) 重试请求模型（若 run_chatgpt 可重试），2) raise 更明确的错误
            # 下面示例选择抛出带有调试输出的异常（便于 CI / 手动调试）
            raise ValueError(f"Unable to parse JSON from model output. Last candidate: {used_candidate}")
        else:
            output = parsed

    agent_events = output

        # ---------- begin patch: normalize event fields ----------
    def normalize_event_fields(e):
        """
        标准化一条 event 的字段名和格式，返回新的 dict（不修改原始对象）。
        目标字段名： 'id', 'date' (string), 'sub_event' (string), 'caused_by' (list)
        兼容输入中的 'sub-event', 'sub_event', 'time' 等多种写法。
        """
        new = {}
        # copy original keys but normalize names: lower-case, replace '-' with '_'
        for k, v in e.items():
            nk = k.strip().lower().replace('-', '_')
            new[nk] = v

        # ensure id exists and is string
        if 'id' in new:
            new['id'] = str(new['id']).strip()
        else:
            # if no id, try to construct one (shouldn't normally happen)
            new['id'] = None

        # unify date: prefer 'date', else 'time'
        if 'date' in new and new['date'] is not None:
            new['date'] = str(new['date']).strip()
        elif 'time' in new and new['time'] is not None:
            new['date'] = str(new['time']).strip()
            # optionally remove 'time' key to avoid duplication
            new.pop('time', None)
        else:
            # leave as empty string to avoid KeyError later
            new['date'] = ""

        # unify sub_event text
        if 'sub_event' in new and new['sub_event'] is not None:
            new['sub_event'] = str(new['sub_event']).strip()
        elif 'sub_event' not in new and 'sub_event' in new:
            new['sub_event'] = str(new.get('sub-event', "")).strip()
        else:
            # try original keys
            new['sub_event'] = str(new.get('sub-event', new.get('sub_event', ""))).strip()

        # ensure caused_by is a list
        cb = new.get('caused_by', [])
        if cb is None:
            cb = []
        if isinstance(cb, str):
            # try to parse comma-separated ids
            cb_list = re.findall(r'E\d+', cb)
            new['caused_by'] = cb_list
        elif isinstance(cb, list):
            # strip items
            new['caused_by'] = [str(x).strip() for x in cb]
        else:
            new['caused_by'] = []

        return new

    # normalize all agent_events right after initial parse
    agent_events = [normalize_event_fields(e) for e in agent_events]

    # print for debug (optional)
    print("Normalized initial events (sample):")
    for e in agent_events[:10]:
        print(e)
    # ---------- end patch ----------

    print("The following events have been generated in the initialization step:")
    for e in agent_events:
        print(list(e.items()))

    # Step 2: continue generation
    while len(agent_events) < args.num_events:
        logging.info("Generating next set of events; current tally = %s" % len(agent_events))
        last_event_id = agent_events[-1]["id"]
        next_event_ids = ['E' + str(i) for i in list(range(int(last_event_id[1:]) + 1, int(last_event_id[1:]) + 5))]
        next_event_id_string = ', '.join(next_event_ids[:3]) + ' and ' + next_event_ids[-1] 
        query = EVENT_KG_FROM_PERSONA_PROMPT_SEQUENTIAL_CONTINUE % (persona_examples[0], 
                                                                   json.dumps(task['examples'][0]["output"][:12], indent=2),
                                                                   next_event_id_string,
                                                                   input,
                                                                   json.dumps(agent_events, indent=2)
                                                                   )
        query_length = num_tokens_from_string(query, 'gpt-3.5-turbo')
        request_length = min(1024, 4096-query_length)
        try:
            output = run_chatgpt(query, num_gen=1, num_tokens_request=request_length, use_16k=False, temperature=1.0).strip()
            output = json.loads(output)
        except:
            output = run_chatgpt(query, num_gen=1, num_tokens_request=request_length, use_16k=False, temperature=1.0).strip()
            output = json.loads(output)
        
        existing_eids = [e["id"] for e in agent_events]
        agent_events.extend([o for o in output if o["id"] not in existing_eids])
        print("Adding events:")
        for e in agent_events:
            print(list(e.items()))

        # filter out standalone events
        if len(agent_events) > args.num_events:
            agent_events = filter_events(agent_events)

    agent_events = renumber_events(agent_events)
    return agent_events


def filter_events(events):

    id2events = {e["id"]: e for e in events}
    remove_ids = []
    for id in id2events.keys():
        # print(id)
        has_child = False
        # check if event has parent
        if len(id2events[id]["caused_by"]) > 0:
            continue
        # check if event has children
        for e in events:
            if id in e["caused_by"]:
                # print("Found %s in %s" % (id, e['id']))
                has_child = True
        
        if not has_child:
            # print("Did not find any connections for %s" % id)
            remove_ids.append(id)
    
    print("*** Removing %s standalone events from %s events: %s ***" % (len(remove_ids), len(id2events), ', '.join(remove_ids)))
    # for id in remove_ids:
        # print(id2events[id])
    
    return [e for e in events if e["id"] not in remove_ids]


def renumber_events(agent_events, prefix='E'):
    """
    1) 按日期（或按出现顺序）对 events 做稳定排序（这里按原列表顺序保序）；
    2) 为每个事件生成新的连续 id: E1..EN；
    3) 返回新的事件列表，并把 caused_by 重映射成新 id。
    """
    # build old_id -> event mapping
    old_ids = [e.get('id') for e in agent_events]
    # Create mapping old->new
    id_map = {}
    for new_idx, old_id in enumerate(old_ids, start=1):
        id_map[old_id] = f"{prefix}{new_idx}"

    # Apply mapping to events
    new_events = []
    for e in agent_events:
        new_e = dict(e)  # shallow copy
        old_id = e.get('id')
        new_e['id'] = id_map.get(old_id, old_id)

        # remap caused_by list
        if 'caused_by' in new_e and isinstance(new_e['caused_by'], list):
            new_caused = []
            for cid in new_e['caused_by']:
                new_caused.append(id_map.get(cid, cid))  # keep old if mapping missing
            new_e['caused_by'] = new_caused
        new_events.append(new_e)
    return new_events

# 用法：在 get_events 的 return 之前
# agent_events = renumber_events(agent_events)
# return agent_events



