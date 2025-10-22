#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_organizer.py

将任意格式（类似用户示例）的 JSON 转换为目标结构：
{
  "conversation": { "speaker_a": ..., "speaker_b": ..., "session_1_date_time": ..., "session_1": [...], ... },
  "event_summary": { "events_session_1": {speakerA: [...], speakerB: [...], "date": ...}, ... },
  "session_summary": { "session_1_summary": "...", ... }
}

用法示例:
  python data_organizer.py --input tom.json --output transformed_output.json
  python data_organizer.py --input driver.json --output Sara.json --speaker-a Sara --speaker-b CarBU-Agent
"""

from __future__ import annotations
import json
import re
import os
import argparse
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

def find_session_numbers(data: Dict[str, Any]) -> List[int]:
    nums: Set[int] = set()
    for k in data.keys():
        m = re.search(r'session_(\d+)', k)
        if m:
            nums.add(int(m.group(1)))
    return sorted(nums)

def detect_speakers(data: Dict[str, Any], session_nums: List[int]) -> List[str]:
    speakers: List[str] = []
    for n in session_nums:
        key = f"session_{n}"
        for turn in data.get(key, []) or []:
            sp = turn.get("speaker")
            if sp and sp not in speakers:
                speakers.append(sp)
            if len(speakers) >= 2:
                return speakers[:2]
    # fallback candidates
    if "name" in data and data["name"] not in speakers:
        speakers.append(data["name"])
    # scan text for common agent tokens
    fallback_tokens = ["CarBU-Agent", "Car-Agent", "Agent", "Caroline", "Melanie"]
    text_blob = json.dumps(data, ensure_ascii=False)
    for tok in fallback_tokens:
        if tok not in speakers and tok in text_blob:
            speakers.append(tok)
        if len(speakers) >= 2:
            break
    # guarantee at least two entries (may be same)
    if not speakers:
        speakers = ["", ""]
    elif len(speakers) == 1:
        speakers.append(speakers[0])
    return speakers[:2]

def extract_date_from_datetime_str(dt_str: Optional[str]) -> Optional[str]:
    """
    给定类似 "9:44 am on 24 April, 2022" 的字符串，尝试返回日期部分 "24 April, 2022"。
    如果无法解析，返回原始字符串或 None。
    """
    if not dt_str:
        return None
    # 优先匹配 "on <date>" 模式
    m = re.search(r'on\s+(.+)$', dt_str)
    if m:
        return m.group(1).strip()
    # 尝试只取逗号后的年份附近
    # 若简单形式 "24 April, 2022" 直接返回
    if re.search(r'\d{4}', dt_str):
        return dt_str.strip()
    return dt_str.strip()

def merge_event_entries(ev_list: Any, default_speakers: List[str]) -> Dict[str, Any]:
    """
    输入可能是 [] 或 [{speaker: [...], ... , "date": "..."}] 等形式。
    将其合并为 { speakerA: [...], speakerB: [...], "date": ... }
    """
    merged: Dict[str, List[str]] = {}
    date_val: Optional[str] = None

    if isinstance(ev_list, list) and ev_list:
        # 合并多个 dict 条目
        for entry in ev_list:
            if not isinstance(entry, dict):
                continue
            # entry 可能包含 "date" 或者每个 speaker 的列表
            if "date" in entry:
                date_val = entry.get("date") or date_val
            for k, v in entry.items():
                if k == "date":
                    continue
                if k not in merged:
                    merged[k] = []
                if isinstance(v, list):
                    merged[k].extend([str(x) for x in v])
                elif isinstance(v, str):
                    merged[k].append(v)
    # 若没有条目或合并后空，创建默认的空数组 for default_speakers
    if not merged:
        for s in default_speakers:
            merged[s] = []
    else:
        # 保证默认 speakers 存在于 merged（避免缺失）
        for s in default_speakers:
            merged.setdefault(s, [])

    # 去重并保持顺序
    for k, lst in list(merged.items()):
        seen = set()
        deduped = []
        for item in lst:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        merged[k] = deduped

    return {"merged": merged, "date": date_val}

def transform(data: Dict[str, Any], forced_speaker_a: Optional[str]=None, forced_speaker_b: Optional[str]=None) -> Dict[str, Any]:
    session_nums = find_session_numbers(data)
    speaker_a, speaker_b = ("", "")
    if forced_speaker_a or forced_speaker_b:
        speaker_a = forced_speaker_a or ""
        speaker_b = forced_speaker_b or ""
    else:
        detected = detect_speakers(data, session_nums)
        speaker_a, speaker_b = detected[0], detected[1]

    # Build conversation block
    conversation = OrderedDict()
    conversation["speaker_a"] = speaker_a
    conversation["speaker_b"] = speaker_b

    for n in session_nums:
        dt_key = f"session_{n}_date_time"
        sess_key = f"session_{n}"
        conversation[dt_key] = data.get(dt_key) if dt_key in data else None
        conversation[sess_key] = data.get(sess_key, []) or []

    # Build event_summary block
    event_summary = OrderedDict()
    for n in session_nums:
        ev_key = f"events_session_{n}"
        ev_raw = data.get(ev_key, None)
        merged_info = merge_event_entries(ev_raw, [conversation["speaker_a"], conversation["speaker_b"]])
        merged = merged_info["merged"]
        date_val = merged_info["date"]

        # 如果 date 为空，尝试从 conversation 的 date_time 字段中提取
        if not date_val:
            date_time_str = conversation.get(f"session_{n}_date_time")
            date_val = extract_date_from_datetime_str(date_time_str)

        # 把 date 插入到 merged dict
        merged_with_date = dict(merged)
        merged_with_date["date"] = date_val
        event_summary[f"events_session_{n}"] = merged_with_date

    # Build session_summary block
    session_summary = OrderedDict()
    for n in session_nums:
        key = f"session_{n}_summary"
        session_summary[key] = data.get(key, "")

    final = OrderedDict()
    final["conversation"] = conversation
    final["event_summary"] = event_summary
    final["session_summary"] = session_summary
    return final

def main():
    p = argparse.ArgumentParser(description="Transform session JSON into target schema.")
    p.add_argument("--input", "-i", required=True, help="输入 JSON 文件路径")
    p.add_argument("--output", "-o", required=True, help="输出 JSON 文件路径")
    p.add_argument("--speaker-a", help="强制指定 conversation.speaker_a")
    p.add_argument("--speaker-b", help="强制指定 conversation.speaker_b")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(base_dir + "/data", args.input)
    output_path = os.path.join(base_dir + "/data", args.output)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transformed = transform(data, forced_speaker_a=args.speaker_a, forced_speaker_b=args.speaker_b)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformed, f, indent=2, ensure_ascii=False)

    print(f"✅ 转换完成：已写入 {args.output}")

if __name__ == "__main__":
    main()
