#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qa_generator.py

从每个 session 的对话 (session_X 字段) 生成 QA，并把证据映射为对话中的 dia_id。

用法:
  python qa_generator.py -i input.json -o qa.json -k 2
  k: 每个 session 生成多少 QA 对
"""

import os
import re
import json
import time
import tqdm
import argparse
import logging
from typing import Any, Dict, List, Optional

# 尝试 new OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
    import openai as _openai_old  # type: ignore

logging.basicConfig(level=logging.INFO)


os.environ["OPENAI_API_KEY"] = "sk-1Bd3OrhU3eg3S2wY5yKwerOlcpu4HuEXAerM8S7ybT4GVKJj"
os.environ["OPENAI_BASE_URL"] = "https://ai.nengyongai.cn/v1"
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"



def make_openai_client_from_env() -> Any:
    api_key = os.environ.get("OPENAI_API_KEY", None)
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    if OpenAI is not None:
        return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    else:
        _openai_old.api_key = api_key
        if base_url:
            _openai_old.api_base = base_url
        return _openai_old


def extract_sessions_from_file(data: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    sessions = {}
    # 首先检查 data["conversation"]
    conv_root = data.get("conversation", data)
    for k, v in conv_root.items():
        m = re.match(r'session_(\d+)$', k)
        if m and isinstance(v, list):
            sessions[k] = v
    # 若为空，再检查顶层
    if not sessions:
        for k, v in data.items():
            m = re.match(r'session_(\d+)$', k)
            if m and isinstance(v, list):
                sessions[k] = v
    return dict(sorted(sessions.items(), key=lambda x: int(re.search(r'(\d+)', x[0]).group(1))))


def session_dialog_to_prompt_block(session_dialog: List[Dict[str, str]]) -> str:
    lines = []
    for turn in session_dialog:
        dia = turn.get("dia_id", "").strip()
        sp = turn.get("speaker", "").strip()
        txt = turn.get("text", "").replace("\n", " ").strip()
        lines.append(f"[{dia}] {sp}: {txt}")
    return "\n".join(lines)


def build_prompt_for_dialogue(dialog_block: str, n_q: int = 2) -> str:
    prompt = f"""
You are a strict QA extraction assistant. Given the dialogue lines below (each line includes a dia_id like [D1:1]),
produce exactly {n_q} question-answer pairs that are strictly grounded in the dialogue between driver and CarAgent,
questions should focus on driver’s factual or behavioral detail derived from the dialogue.
Requirements:
1) Use ONLY the information present in the dialogue below. Do NOT invent or infer facts.
2) Each question must be a wh- question (between these 4 types:Who/What/When/Where).
3) For each QA pair, provide:
   - "question": the question text,
   - "answer": a concise answer strictly supported by the dialogue,
   - "evidence": a JSON array of the dia_id strings (e.g., ["D1:1","D1:3"]) that directly support the answer.
4) If the answer cannot be determined from the dialogue, set the answer to exactly "Not stated in the text." and evidence to [].
5) Output EXACTLY one JSON array (no extra commentary) with {n_q} objects of the shape:
   [{{"question":"...","answer":"...","evidence":["D1:1"]}}, ...]
6) Do not include anything else outside the JSON array.
7) Use third person perspective when generating questions.

Here is the dialogue (do NOT use anything outside this block):
===BEGIN DIALOGUE===
{dialog_block}
===END DIALOGUE===

Now produce the required JSON array.
""".strip()
    return prompt



def parse_json_from_text(text: str) -> Optional[Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'(\[.*\])', text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r'(\{.*\})', text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


def normalize_text(t: str) -> str:
    return re.sub(r'\s+', ' ', t).strip().lower()


def find_evidence_by_answer(answer: str, session_dialog: List[Dict[str, str]], top_k: int = 3) -> List[str]:
    """
    基于 answer 文本在对话中检索最匹配的 turns，并返回其 dia_id 列表（最多 top_k）。
    用作模型给出的 evidence 无效时的 fallback。
    """
    ans_norm = normalize_text(answer)
    if not ans_norm or ans_norm == "not stated in the text.":
        return []
    scores = []
    for turn in session_dialog:
        txt = normalize_text(turn.get("text", ""))
        if not txt:
            continue
        # 简单相似度：包含关键短语、词重叠数
        overlap = sum(1 for w in ans_norm.split() if len(w) > 2 and w in txt)
        # 额外加权：若连续子串出现
        common_seq = 1 if ans_norm in txt else 0
        score = overlap + common_seq
        if score > 0:
            scores.append((score, turn.get("dia_id")))
    scores.sort(key=lambda x: x[0], reverse=True)
    # 返回去重的 dia_id
    seen = set()
    out = []
    for _, did in scores:
        if did and did not in seen:
            out.append(did)
            seen.add(did)
            if len(out) >= top_k:
                break
    return out


def validate_and_fix_qas(parsed: Any, valid_ids: set, session_dialog: List[Dict[str, str]], n_q: int) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(parsed, list):
        return out
    for i in range(n_q):
        if i >= len(parsed):
            break
        item = parsed[i]
        if not isinstance(item, dict):
            continue
        q = str(item.get("question") or item.get("q") or "").strip()
        a = str(item.get("answer") or item.get("a") or "").strip()
        evidence = item.get("evidence") or item.get("evidences") or item.get("evidence_ids") or []
        if not isinstance(evidence, list):
            if isinstance(evidence, str):
                evidence = re.findall(r'[A-Za-z0-9:_-]+', evidence)
            else:
                evidence = []
        # filter valid dia ids
        evidence_filtered = [e for e in evidence if e in valid_ids]
        # fallback: 若 evidence_filtered 为空且 answer 非 "Not stated..."，尝试检索对话
        if not evidence_filtered and a and a != "Not stated in the text.":
            evidence_filtered = find_evidence_by_answer(a, session_dialog, top_k=3)
            evidence_filtered = [e for e in evidence_filtered if e in valid_ids]
        out.append({"question": q or "Not stated in the text.", "answer": a or "Not stated in the text.", "evidence": evidence_filtered})
    return out


def generate_qas_for_session(client: Any, dialog_block: str, session_dialog: List[Dict[str, str]], model: str, n_q: int = 2, max_retries: int = 3) -> List[Dict[str, Any]]:
    prompt = build_prompt_for_dialogue(dialog_block, n_q=n_q)
    for attempt in range(1, max_retries + 1):
        try:
            if hasattr(client, "chat"):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a strict QA extraction assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=1024
                )
                try:
                    model_text = resp['choices'][0]['message']['content']
                except Exception:
                    model_text = resp.choices[0].message.content  # type: ignore
            else:
                resp = client.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a strict QA extraction assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=1024
                )
                model_text = resp['choices'][0]['message']['content']

            parsed = parse_json_from_text(model_text)
            if parsed is None:
                logging.warning("Attempt %s: couldn't parse JSON. Raw:\n%s", attempt, model_text)
                raise ValueError("parse failed")

            valid_ids = set([t.get("dia_id") for t in session_dialog if t.get("dia_id")])
            qas = validate_and_fix_qas(parsed, valid_ids, session_dialog, n_q)

            # 如果某些 QA 的 answer 非 "Not stated..." 但 evidence 为空，则重试一次
            evidence_missing = any((item["answer"] and item["answer"] != "Not stated in the text." and len(item["evidence"]) == 0) for item in qas)
            if evidence_missing and attempt < max_retries:
                logging.info("Some QA lack evidence - retrying (attempt %s)...", attempt+1)
                time.sleep(0.5 * attempt)
                continue

            # 补全不足 q 的条目
            while len(qas) < n_q:
                qas.append({"question": "Fallback question (could not generate)", "answer": "Not stated in the text.", "evidence": []})

            return qas[:n_q]
        except Exception as e:
            logging.warning("Model attempt %s failed: %s", attempt, e)
            time.sleep(0.5 * attempt)
            continue

    # 全部失败，返回 fallback
    return [{"question": f"Fallback question {i+1}", "answer": "Not stated in the text.", "evidence": []} for i in range(n_q)]


def safe_model_generate_for_session(client, dialog_block, session_dialog, model, n_q=2, max_retries=3):
    """
    将原来的 generate_qas_for_session 包装成单 session 安全调用（同之前脚本）。
    返回 list of {"question","answer","evidence": [...]}
    """
    # 直接复用你已有的函数名。如果你已经有 generate_qas_for_session，请调用它
    return generate_qas_for_session(client, dialog_block, session_dialog, model=model, n_q=n_q, max_retries=max_retries)


def main():
    # （省略参数解析与 client 创建，沿用你已有的 args, client）
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--per-session", "-k", type=int, default=2)
    parser.add_argument("--model", "-m", type=str, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    client = make_openai_client_from_env()
    model = args.model or os.environ.get("OPENAI_MODEL_NAME")
    if not model:
        raise ValueError("No model specified")


    input_path = os.path.join("/workspace/dataset/data", args.input)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions = extract_sessions_from_file(data)
    if not sessions:
        raise ValueError("No sessions found")

    output_path = os.path.join("/workspace/dataset/data", args.output)
    # 输出文件若已存在，则增量写入（避免覆盖）；先载入已有结果
    if os.path.exists(output_path):
        out_data = json.load(open(output_path, "r", encoding="utf-8"))
    else:
        out_data = {}

    
    all_qas = []

    for s_key, s_dialog in sessions.items():
        logging.info("Processing %s (%d turns)...", s_key, len(s_dialog))
        dialog_block = session_dialog_to_prompt_block(s_dialog)

        qas = safe_model_generate_for_session(
            client, dialog_block, s_dialog, model=model,
            n_q=args.per_session, max_retries=args.max_retries
        )

        # 直接扩展到全局列表
        all_qas.extend(qas)

        # 可选：每完成一个 session 即刻保存，防止中断丢失
        output_path = os.path.join("/workspace/dataset/data", args.output)
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump({"qa": all_qas}, fout, indent=2, ensure_ascii=False)

        time.sleep(0.3)

    logging.info("所有 session QA 已生成并合并完成。")

    
if __name__ == "__main__":
    main()