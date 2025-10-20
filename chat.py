import os
import json
import time
import warnings
import logging
from urllib3.exceptions import InsecureRequestWarning
import httpx
import openai
from openai import OpenAI
import prompts

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("car_data_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== 配置区 =====================
OPENAI_API_KEY = "sk-1Bd3OrhU3eg3S2wY5yKwerOlcpu4HuEXAerM8S7ybT4GVKJj"
OUTFILE = "/workspace/dataset/car_test.json"
MODEL_NAME = "gpt-5-2025-08-07"

# 忽略不安全请求警告
warnings.simplefilter("ignore", InsecureRequestWarning)

# ===================== 初始化 HTTP 客户端 =====================
logger.info("初始化HTTP客户端")
try:
    httpx_client = httpx.Client(
        timeout=httpx.Timeout(connect=60.0, read=300.0, write=60.0, pool=30.0),
        verify=False,              # 绕过SSL验证（仅限受信环境）
        follow_redirects=True      # 启用302自动跳转
    )
    logger.info("HTTP客户端初始化成功（已启用自动重定向）")
except Exception as e:
    logger.error(f"HTTP客户端初始化失败: {str(e)}", exc_info=True)
    raise

# ===================== 初始化 OpenAI 客户端 =====================
logger.info("初始化OpenAI客户端")
try:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://ai.nengyongai.cn/v1",  # 如使用官方接口改成 https://api.openai.com/v1
        http_client=httpx_client
    )
    logger.info("OpenAI客户端初始化成功")
except Exception as e:
    logger.error(f"OpenAI客户端初始化失败: {str(e)}", exc_info=True)
    raise

# ===================== 提示词 =====================
dataset_prompt = prompts.multiple_characters_prompt

# ===================== 自动分段生成函数 =====================
def generate_large_output(client, model, prompt, max_tokens_per_call=4000, max_parts=30, temperature=0.6):
    """
    自动分段生成大文本，直到模型输出 [END] 或达到 max_parts。
    每段输出约 max_tokens_per_call tokens。
    """
    all_text = ""
    for part in range(1, max_parts + 1):
        logger.info(f"生成第 {part} 段内容 ...")

        sub_prompt = f"{prompt}\n\n请生成第 {part} 段（每段约 {max_tokens_per_call} tokens）。" \
                     f"若已生成全部内容，请在末尾输出 [END]。"

        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": sub_prompt}],
                max_tokens=max_tokens_per_call,
                temperature=temperature,
                stop=None,
                timeout=300
            )
            elapsed_time = time.time() - start_time
            text = response.choices[0].message.content.strip()
            logger.info(f"第 {part} 段生成成功，长度: {len(text)} 字符，耗时 {elapsed_time:.2f} 秒")

            all_text += text + "\n"

            if "[END]" in text:
                logger.info("检测到 [END]，停止生成")
                break

        except Exception as e:
            logger.error(f"第 {part} 段生成失败: {str(e)}", exc_info=True)
            break

        time.sleep(1.0)  # 防止触发速率限制

    return all_text.strip()

# ===================== 主执行逻辑 =====================
logger.info("开始生成长文本输出")
try:
    output_text = generate_large_output(
        client=client,
        model=MODEL_NAME,
        prompt=dataset_prompt,
        max_tokens_per_call=4000,  # 每次调用最多生成4000 tokens
        max_parts=30,              # 最多分30段
        temperature=0.6
    )
    logger.info(f"生成完成，总长度: {len(output_text)} 字符")
except Exception as e:
    logger.error(f"生成输出失败: {str(e)}", exc_info=True)
    raise

# ===================== 保存结果 =====================
if not output_text:
    logger.error("输出内容为空，未保存。")
else:
    try:
        json_obj = json.loads(output_text)
        with open(OUTFILE, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存JSON格式数据到 {OUTFILE}")
    except json.JSONDecodeError:
        with open(OUTFILE, "w", encoding="utf-8") as f:
            f.write(output_text)
        logger.warning("输出不是有效JSON，已保存为纯文本。")
    except Exception as e:
        logger.error(f"保存文件失败: {str(e)}", exc_info=True)
        raise

# ===================== 清理资源 =====================
httpx_client.close()
logger.info("程序执行完成 ✅")
