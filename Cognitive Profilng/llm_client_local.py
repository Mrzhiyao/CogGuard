from openai import OpenAI
from typing import Optional
import os
import httpx


class LLMClient:
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = None,
                 model: str = None,
                 timeout: float = 60.0,
                 enable_thinking: bool = True):

        self.api_key = api_key or ""
        self.base_url = base_url or ""
        self.model = model
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        
        timeout_obj = httpx.Timeout(timeout, connect=10.0)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout_obj,
            max_retries=0,  # 禁用 OpenAI SDK 的内置重试
        )
    
    def call_llm(self, prompt: str, temperature: float = 0.3, max_tokens: int = 4000) -> str:
        try:
            # 构建请求参数
            request_params = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if "dashscope.aliyuncs.com" in self.base_url:
                if self.enable_thinking:
                    request_params["stream"] = True
                    request_params["extra_body"] = {
                        "enable_thinking": True
                    }
                    
                    stream = self.client.chat.completions.create(**request_params)
                    response_parts = []
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            response_parts.append(chunk.choices[0].delta.content)
                        # 检查是否完成
                        if chunk.choices[0].finish_reason is not None:
                            break
                    response = "".join(response_parts)
                else:
                    request_params["extra_body"] = {
                        "enable_thinking": False
                    }
                    chat_completion = self.client.chat.completions.create(**request_params)
                    response = chat_completion.choices[0].message.content
            else:
                # 非阿里云 API，使用非流式调用
                chat_completion = self.client.chat.completions.create(**request_params)
                response = chat_completion.choices[0].message.content
            
            return response
        except Exception as e:
            print(f"LLM调用出错: {e}")
            raise
    
    async def call_llm_async(self, prompt: str, temperature: float = 0.3, max_tokens: int = 4000) -> str:

        import asyncio
        # 在线程池中运行同步调用，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.call_llm,
            prompt,
            temperature,
            max_tokens
        )
        return response


# 默认客户端实例（使用默认配置）
_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    """获取默认LLM客户端实例"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client





