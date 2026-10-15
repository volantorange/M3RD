import requests
import json
import os


class Qwentest:
    def __init__(self, api_key=None, base_url="https://dashscope.aliyuncs.com/api/v1"):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key")
        self.base_url = base_url

    def chat_completion(self, model, messages, temperature=0.7, stream=False):
        """直接调用HTTP API"""
        url = f"{self.base_url}/services/aigc/text-generation/generation"

        data = {
            "model": model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": temperature,
                "result_format": "message"
            }
        }


        response = requests.post(
            url,
            json=data,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60,
        )
        return self._handle_response(response)

    def _handle_response(self, response):
        """处理普通响应"""
        if response.status_code == 200:
            result = response.json()
            if "output" in result and "choices" in result["output"]:
                return result["output"]["choices"][0]["message"]["content"]
        print(f"请求失败: {response.status_code} - {response.text}")
        return None


# 使用示例
if __name__ == "__main__":
    client = Qwentest()


    messages = [
        {
            "role": "user",
            "content": "你是谁"
        }
    ]

    # 非流式
    print("=== 非流式调用 ===")
    response = client.chat_completion(
        model="qwen-plus",
        messages=messages,
    )
    if response:
        print(type(response))
