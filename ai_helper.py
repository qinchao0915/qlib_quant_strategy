#!/usr/bin/env python3
"""
AI 助手 - 直接调用百炼 API
替代 Claude Code 用于量化代码开发
"""

from openai import OpenAI
import sys
import os

# 从环境变量读取 API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    # 尝试从 API_KEYS.md 读取
    api_keys_path = "/home/node/.openclaw/workspace/API_KEYS.md"
    try:
        with open(api_keys_path) as f:
            content = f.read()
            # 提取 API Key
            import re
            match = re.search(r'API Key.*`([^`]+)`', content)
            if match:
                DASHSCOPE_API_KEY = match.group(1)
    except Exception:
        pass

if not DASHSCOPE_API_KEY:
    print("❌ 错误: 找不到 DASHSCOPE_API_KEY")
    print("请设置环境变量: export DASHSCOPE_API_KEY=sk-...")
    sys.exit(1)

# 配置百炼 API
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://coding.dashscope.aliyuncs.com/v1"
)


def chat(prompt, model="qwen3.5-plus"):
    """
    与 AI 对话
    
    Args:
        prompt: 提示词
        model: 模型名称
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的量化交易代码助手，擅长 Python、机器学习、金融数据分析。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
    else:
        prompt = input("请输入问题: ")
    
    print("🤖 AI 助手思考中...\n")
    result = chat(prompt)
    print(result)
