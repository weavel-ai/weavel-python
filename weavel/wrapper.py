from dotenv import load_dotenv
import os
from typing import Dict, Optional
from uuid import uuid4
from openai import AsyncOpenAI, OpenAI
from datetime import datetime, timezone
import time

from weavel._worker import Worker

pricing = {
    "gpt-4o": {"input": 0.000005, "output": 0.000015},
    "gpt-4o-2024-08-06": {"input": 0.0000025, "output": 0.00001},
    "gpt-4o-2024-05-13": {"input": 0.000005, "output": 0.000015},
    "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
    "gpt-4o-mini-2024-07-18": {"input": 0.00000015, "output": 0.0000006},
}

DEFAULT_PARAMS = {
    "model": None,
    "frequency_penalty": 0,
    "logit_bias": None,
    "logprobs": False,
    "top_logprobs": None,
    "max_tokens": None,
    "n": 1,
    "presence_penalty": 0,
    "response_format": None,
    "seed": None,
    "service_tier": None,
    "stop": None,
    "stream": False,
    "stream_options": None,
    "temperature": 1,
    "top_p": 1,
    "tools": None,
    "tool_choice": None,
    "parallel_tool_calls": True,
    "user": None,
}

load_dotenv()


def calculate_cost(model: str, usage: Dict[str, int]) -> float:
    if model not in pricing:
        # raise ValueError(f"Unknown model: {model}")
        return round(0, 6)

    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    input_cost = input_tokens * pricing[model]["input"]
    output_cost = output_tokens * pricing[model]["output"]

    total_cost = input_cost + output_cost

    return round(total_cost, 6)


class LoggingSuppressor:
    def __init__(self, worker):
        self.worker = worker
        self.original_capture_generation = None

    def __enter__(self):
        self.original_capture_generation = self.worker.capture_generation
        self.worker.capture_generation = lambda *args, **kwargs: None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.worker.capture_generation = self.original_capture_generation


class AsyncLoggingSuppressor:
    def __init__(self, worker):
        self.worker = worker
        self.original_capture_generation = None
        self.original_acapture_generation = None

    async def __aenter__(self):
        self.original_capture_generation = self.worker.capture_generation
        self.original_acapture_generation = self.worker.acapture_generation
        self.worker.capture_generation = None
        self.worker.acapture_generation = None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.worker.capture_generation = self.original_capture_generation
        self.worker.acapture_generation = self.original_acapture_generation


class WeavelOpenAI(OpenAI):
    def __init__(
        self,
        *args,
        weavel_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weavel_api_key = weavel_api_key or os.getenv("WEAVEL_API_KEY")
        self.original_chat_completions = self.chat.completions
        self.original_beta_chat_completions = self.beta.chat.completions

        class CustomChatCompletions:
            def __init__(self, original, weavel_api_key: str, base_url: Optional[str]):
                self.original = original
                self.weavel_api_key = weavel_api_key
                self.base_url = base_url
                self._worker = Worker(
                    self.weavel_api_key,
                    self.base_url,
                    flush_interval=3,
                )
                self.header = {}

            def create(self, *args, **kwargs):
                self.header = kwargs.pop("headers", {})
                is_streaming = kwargs.get("stream", False)

                if is_streaming:
                    return self._handle_streaming(*args, **kwargs)
                else:
                    return self._handle_non_streaming(*args, **kwargs)

            def _handle_streaming(self, *args, **kwargs):
                kwargs.setdefault("stream_options", {})
                kwargs["stream_options"]["include_usage"] = True

                start_time = time.time()
                response = self.original.create(*args, **kwargs)
                accumulated_response = {
                    "id": None,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "function_call": None,
                                "tool_calls": None,
                            },
                            "finish_reason": None,
                        }
                    ],
                    "created": None,
                    "model": None,
                    "object": "chat.completion",
                    "usage": None,
                    "service_tier": None,
                    "system_fingerprint": None,
                }

                for chunk in response:
                    if len(chunk.choices) == 0:
                        response_details = chunk.model_dump()
                        accumulated_response["id"] = response_details["id"]
                        accumulated_response["created"] = response_details["created"]
                        accumulated_response["model"] = response_details["model"]
                        accumulated_response["object"] = response_details["object"]
                        accumulated_response["usage"] = response_details["usage"]
                        accumulated_response["service_tier"] = response_details[
                            "service_tier"
                        ]
                        accumulated_response["system_fingerprint"] = response_details[
                            "system_fingerprint"
                        ]
                        continue
                    else:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            accumulated_response["choices"][0]["message"][
                                "content"
                            ] += delta.content
                        if delta.function_call:
                            if (
                                accumulated_response["choices"][0]["message"][
                                    "function_call"
                                ]
                                is None
                            ):
                                accumulated_response["choices"][0]["message"][
                                    "function_call"
                                ] = {"name": "", "arguments": ""}
                            accumulated_response["choices"][0]["message"][
                                "function_call"
                            ]["name"] += (delta.function_call.name or "")
                            accumulated_response["choices"][0]["message"][
                                "function_call"
                            ]["arguments"] += (delta.function_call.arguments or "")
                        if delta.tool_calls:
                            if (
                                accumulated_response["choices"][0]["message"][
                                    "tool_calls"
                                ]
                                is None
                            ):
                                accumulated_response["choices"][0]["message"][
                                    "tool_calls"
                                ] = []
                            accumulated_response["choices"][0]["message"][
                                "tool_calls"
                            ].extend(delta.tool_calls)
                        if chunk.choices[0].finish_reason:
                            accumulated_response["choices"][0]["finish_reason"] = (
                                chunk.choices[0].finish_reason
                            )

                    yield chunk

                end_time = time.time()
                latency = end_time - start_time
                accumulated_response["latency"] = round(latency, 6)

                model = accumulated_response["model"]
                usage = accumulated_response["usage"]
                cost = calculate_cost(model, usage)
                accumulated_response["cost"] = cost

                self._capture_generation(inputs=kwargs, outputs=accumulated_response)

            def _handle_non_streaming(self, *args, **kwargs):
                start_time = time.time()
                response = self.original.create(*args, **kwargs)
                end_time = time.time()
                latency = end_time - start_time
                latency = round(latency, 6)

                formatted_response = {
                    "id": response.id,
                    "choices": [choice.model_dump() for choice in response.choices],
                    "created": response.created,
                    "model": response.model,
                    "object": response.object,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "service_tier": response.service_tier,
                    "system_fingerprint": response.system_fingerprint,
                    "latency": latency,
                }

                # Calculate and add cost
                cost = calculate_cost(response.model, formatted_response["usage"])
                formatted_response["cost"] = cost

                # TODO: models, latency, cost shouldn't be logged as inputs. Should have separate fields for each.
                self._capture_generation(inputs=kwargs, outputs=formatted_response)
                return response

            def _capture_generation(
                self,
                inputs,
                outputs,
            ):
                self._worker.capture_generation(
                    observation_id=str(uuid4()),
                    created_at=datetime.now(timezone.utc),
                    name=self.header.get("name", "OpenAI Chat"),
                    prompt_name=self.header.get("prompt_name", None),
                    inputs=inputs,
                    outputs=outputs,
                )

        class CustomBetaChatCompletions:
            def __init__(self, original, weavel_api_key: str, base_url: Optional[str]):
                self.original = original
                self.weavel_api_key = weavel_api_key
                self.base_url = base_url
                self._worker = Worker(
                    self.weavel_api_key,
                    self.base_url,
                    flush_interval=3,
                )

            def parse(self, *args, **kwargs):
                header = kwargs.pop("headers", {})
                start_time = time.time()

                # context manager to suppress duplicate logging
                with LoggingSuppressor(self._worker):
                    response = self.original.parse(*args, **kwargs)

                end_time = time.time()
                latency = end_time - start_time
                latency = round(latency, 6)

                formatted_response = {
                    "id": response.id,
                    "choices": [choice.model_dump() for choice in response.choices],
                    "created": response.created,
                    "model": response.model,
                    "object": response.object,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "service_tier": response.service_tier,
                    "system_fingerprint": response.system_fingerprint,
                    "latency": latency,
                }

                # Calculate and add cost
                cost = calculate_cost(response.model, formatted_response["usage"])
                formatted_response["cost"] = cost

                # TODO: models, latency, cost shouldn't be logged as inputs. Should have separate fields for each.
                self._worker.capture_generation(
                    observation_id=str(uuid4()),
                    created_at=datetime.now(timezone.utc),
                    name=header.get("name", "OpenAI Beta Chat Parse"),
                    prompt_name=header.get("prompt_name", None),
                    inputs=kwargs,
                    outputs=formatted_response,
                )

                return response

        self.chat.completions = CustomChatCompletions(
            self.original_chat_completions, self.weavel_api_key, base_url
        )
        self.beta.chat.completions = CustomBetaChatCompletions(
            self.original_beta_chat_completions, self.weavel_api_key, base_url
        )


class AsyncWeavelOpenAI(AsyncOpenAI):
    def __init__(
        self,
        *args,
        weavel_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weavel_api_key = weavel_api_key or os.getenv("WEAVEL_API_KEY")
        self.original_chat_completions = self.chat.completions
        self.original_beta_chat_completions = self.beta.chat.completions

        class AsyncCustomChatCompletions:
            def __init__(self, original, weavel_api_key: str, base_url: Optional[str]):
                self.original = original
                self.weavel_api_key = weavel_api_key
                self.base_url = base_url
                self._worker = Worker(
                    self.weavel_api_key,
                    self.base_url,
                    flush_interval=3,
                )
                self.header = {}

            async def create(self, *args, **kwargs):
                self.header = kwargs.pop("headers", {})
                is_streaming = kwargs.get("stream", False)

                if is_streaming:
                    return self._handle_streaming(*args, **kwargs)
                else:
                    return await self._handle_non_streaming(*args, **kwargs)

            async def _handle_streaming(self, *args, **kwargs):
                kwargs.setdefault("stream_options", {})
                kwargs["stream_options"]["include_usage"] = True

                start_time = time.time()
                response = await self.original.create(*args, **kwargs)
                accumulated_response = {
                    "id": None,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "function_call": None,
                                "tool_calls": None,
                            },
                            "finish_reason": None,
                        }
                    ],
                    "created": None,
                    "model": None,
                    "object": "chat.completion",
                    "usage": None,
                    "service_tier": None,
                    "system_fingerprint": None,
                }

                async for chunk in response:
                    if len(chunk.choices) == 0:
                        response_details = chunk.model_dump()
                        accumulated_response["id"] = response_details["id"]
                        accumulated_response["created"] = response_details["created"]
                        accumulated_response["model"] = response_details["model"]
                        accumulated_response["object"] = response_details["object"]
                        accumulated_response["usage"] = response_details["usage"]
                        accumulated_response["service_tier"] = response_details[
                            "service_tier"
                        ]
                        accumulated_response["system_fingerprint"] = response_details[
                            "system_fingerprint"
                        ]
                        continue
                    else:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            accumulated_response["choices"][0]["message"][
                                "content"
                            ] += delta.content
                        if delta.function_call:
                            if (
                                accumulated_response["choices"][0]["message"][
                                    "function_call"
                                ]
                                is None
                            ):
                                accumulated_response["choices"][0]["message"][
                                    "function_call"
                                ] = {"name": "", "arguments": ""}
                            accumulated_response["choices"][0]["message"][
                                "function_call"
                            ]["name"] += (delta.function_call.name or "")
                            accumulated_response["choices"][0]["message"][
                                "function_call"
                            ]["arguments"] += (delta.function_call.arguments or "")
                        if delta.tool_calls:
                            if (
                                accumulated_response["choices"][0]["message"][
                                    "tool_calls"
                                ]
                                is None
                            ):
                                accumulated_response["choices"][0]["message"][
                                    "tool_calls"
                                ] = []
                            accumulated_response["choices"][0]["message"][
                                "tool_calls"
                            ].extend(delta.tool_calls)
                        if chunk.choices[0].finish_reason:
                            accumulated_response["choices"][0]["finish_reason"] = (
                                chunk.choices[0].finish_reason
                            )

                    yield chunk

                end_time = time.time()
                latency = end_time - start_time
                accumulated_response["latency"] = round(latency, 6)

                model = accumulated_response["model"]
                usage = accumulated_response["usage"]
                cost = calculate_cost(model, usage)
                accumulated_response["cost"] = cost

                if self._worker.capture_generation is not None:
                    await self._capture_generation(
                        inputs=kwargs, outputs=accumulated_response
                    )

            async def _handle_non_streaming(self, *args, **kwargs):
                start_time = time.time()
                response = await self.original.create(*args, **kwargs)
                end_time = time.time()
                latency = end_time - start_time
                latency = round(latency, 6)

                formatted_response = {
                    "id": response.id,
                    "choices": [choice.model_dump() for choice in response.choices],
                    "created": response.created,
                    "model": response.model,
                    "object": response.object,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "service_tier": response.service_tier,
                    "system_fingerprint": response.system_fingerprint,
                    "latency": latency,
                }

                # Calculate and add cost
                cost = calculate_cost(response.model, formatted_response["usage"])
                formatted_response["cost"] = cost

                if self._worker.capture_generation is None:
                    return response

                await self._capture_generation(
                    inputs=kwargs, outputs=formatted_response
                )
                return response

            async def _capture_generation(
                self,
                inputs,
                outputs,
            ):
                await self._worker.acapture_generation(
                    observation_id=str(uuid4()),
                    created_at=datetime.now(timezone.utc),
                    name=self.header.get("name", "AsyncOpenAI Chat"),
                    prompt_name=self.header.get("prompt_name", None),
                    inputs=inputs,
                    outputs=outputs,
                )

        class AsyncCustomBetaChatCompletions:
            def __init__(self, original, weavel_api_key: str, base_url: Optional[str]):
                self.original = original
                self.weavel_api_key = weavel_api_key
                self.base_url = base_url
                self._worker = Worker(
                    self.weavel_api_key,
                    self.base_url,
                    flush_interval=3,
                )

            async def parse(self, *args, **kwargs):
                header = kwargs.pop("headers", {})
                start_time = time.time()

                async with AsyncLoggingSuppressor(self._worker):
                    response = await self.original.parse(*args, **kwargs)

                end_time = time.time()
                latency = end_time - start_time
                latency = round(latency, 6)

                formatted_response = {
                    "id": response.id,
                    "choices": [choice.model_dump() for choice in response.choices],
                    "created": response.created,
                    "model": response.model,
                    "object": response.object,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "service_tier": response.service_tier,
                    "system_fingerprint": response.system_fingerprint,
                    "latency": latency,
                }

                # Calculate and add cost
                cost = calculate_cost(response.model, formatted_response["usage"])
                formatted_response["cost"] = cost

                await self._worker.acapture_generation(
                    observation_id=str(uuid4()),
                    created_at=datetime.now(timezone.utc),
                    name=header.get("name", "Async OpenAI Beta Chat Parse"),
                    prompt_name=header.get("prompt_name", None),
                    inputs=kwargs,
                    outputs=formatted_response,
                )
                return response

        self.chat.completions = AsyncCustomChatCompletions(
            self.original_chat_completions, self.weavel_api_key, base_url
        )
        self.beta.chat.completions = AsyncCustomBetaChatCompletions(
            self.original_beta_chat_completions, self.weavel_api_key, base_url
        )
