# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import abc
import json
import multiprocessing
import os
import re
import sys
import time
import requests
import traceback
from pathlib import Path
from typing import List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 


class Client(abc.ABC):
    def __init__(
        self,
        server_host,
        server_port='5000',
        proxy_url=None,
        ssh_server=None,
        ssh_key_path=None,
        **generation_kwargs
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.proxy_url = proxy_url
        self.ssh_server = os.getenv("SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("SSH_KEY_PATH", ssh_key_path)
        self.proxies = None
        if self.proxy_url:
            self.proxies = {
                "http": self.proxy_url,
                "https": self.proxy_url,
            }
        self.generation_kwargs = generation_kwargs
        
    @abc.abstractmethod
    def _single_call(
        self,
        prompts,
    ):
        pass

    def __call__(
        self,
        prompt: str,
        **kwargs
    ):
        request = self.generation_kwargs
        # prompts are added later
        request['prompts'] = [f'{prompt}']
        if 'others' in kwargs:
            request['others'] = kwargs['others']

        outputs = self._single_call(**request)
        response = {'text': outputs}
        return response
        
    @retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(3))
    def _send_request(self, request, route="generate"):
        timeout = self.generation_kwargs.get("request_timeout_seconds", 120)
        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            sshtunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
            outputs = sshtunnel_request.put(
                url="http://{}:{}/{}".format(self.server_host, self.server_port, route),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
            ).json()
        else:
            outputs = requests.put(
                url="http://{}:{}/{}".format(self.server_host, self.server_port, route),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
                proxies=self.proxies,
                timeout=timeout,
            ).json()
        return outputs

    def process_batch(self, prompts: List[str], on_result=None, **kwargs) -> List[dict]:
        num_threads = max(96, multiprocessing.cpu_count() * 16)
        with ThreadPoolExecutor(num_threads) as executor:
            futures = {}
            for index, prompt in enumerate(prompts):
                future = executor.submit(
                    self.__call__,
                    prompt,
                    **kwargs,
                )
                futures[future] = index

            rets = [None] * len(prompts)
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                    rets[index] = result
                    if on_result is not None:
                        on_result(index, True, result)
                except Exception as e:
                    failed_result = {
                        '__failed__': True,
                        'text': '',
                        'error': str(e),
                    }
                    rets[index] = failed_result
                    if on_result is not None:
                        on_result(index, False, failed_result)
        return rets


class TRTLLMClient(Client):
    def _single_call(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        random_seed,
        stop: List[str],
        max_attention_window_size=None,
    ):
        request = {
            "prompts": prompts,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            'stop_words_list': ",".join(stop),
        }
        if max_attention_window_size:
            request["max_attention_window_size"] = max_attention_window_size
            
        outputs = self._send_request(request)
        return outputs


class VLLMClient(Client):
    @staticmethod
    def _extract_error_message(outputs):
        if not isinstance(outputs, dict):
            return str(outputs)
        error_obj = outputs.get('error')
        if isinstance(error_obj, dict):
            return error_obj.get('message', str(error_obj))
        if error_obj is not None:
            return str(error_obj)
        return str(outputs)

    @staticmethod
    def _build_headers(api_key, disable_auth_header):
        headers = {"Content-Type": "application/json"}
        if (not disable_auth_header) and api_key and str(api_key).strip() and str(api_key).strip().upper() != "EMPTY":
            headers["Authorization"] = f"Bearer {str(api_key).strip()}"
        return headers

    @staticmethod
    def _parse_json_response(response, url):
        try:
            return response.json()
        except ValueError as e:
            body = response.text if response.text is not None else ""
            body = re.sub(r'\s+', ' ', body).strip()
            body_preview = body[:400] + (" ... [truncated]" if len(body) > 400 else "")
            content_type = response.headers.get("Content-Type", "")
            raise RuntimeError(
                "Failed to parse JSON response from vLLM endpoint. "
                f"status={response.status_code}, content_type='{content_type}', url='{url}', "
                f"body_preview='{body_preview or '<empty>'}', decode_error='{e}'"
            ) from e

    def _single_call(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        random_seed,
        stop: List[str],
        model_name=None,
        vllm_endpoint='native',
        api_base_url=None,
        api_key='EMPTY',
        disable_auth_header=True,
        reasoning_effort=None,
        truncate_prompt_tokens=0,
        request_timeout_seconds=120,
    ):
        if vllm_endpoint in ('completions', 'chat'):
            timeout = self.generation_kwargs.get("request_timeout_seconds", 120)
            if api_base_url:
                base_url = api_base_url.rstrip('/')
            else:
                base_url = f"http://{self.server_host}:{self.server_port}/v1"

            if vllm_endpoint == 'chat':
                url = f"{base_url}/chat/completions"
                request = {
                    "messages": [{"role": "user", "content": prompts[0]}],
                    "max_tokens": tokens_to_generate,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop": stop,
                    "seed": random_seed,
                }
                if truncate_prompt_tokens and truncate_prompt_tokens > 0:
                    request["truncate_prompt_tokens"] = truncate_prompt_tokens
                if model_name:
                    request["model"] = model_name
                if reasoning_effort:
                    request["reasoning_effort"] = reasoning_effort

                response = requests.post(
                    url=url,
                    data=json.dumps(request),
                    headers=self._build_headers(api_key, disable_auth_header),
                    proxies=self.proxies,
                    timeout=timeout,
                )
                outputs = self._parse_json_response(response, url)

                if 'error' in outputs:
                    msg = self._extract_error_message(outputs)
                    if "max_tokens must be at least 1" in msg:
                        raise RuntimeError(
                            f"vLLM rejected request due to context overflow: {msg}. "
                            "Try reducing data --max_seq_length, reducing output tokens, "
                            "or set --truncate_prompt_tokens (e.g., 32768 or your server max context)."
                        )
                    raise RuntimeError(outputs['error'])

                return outputs['choices'][0]['message']['content']

            url = f"{base_url}/completions"
            request = {
                "prompt": prompts[0],
                "max_tokens": tokens_to_generate,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
                "seed": random_seed,
            }
            if truncate_prompt_tokens and truncate_prompt_tokens > 0:
                request["truncate_prompt_tokens"] = truncate_prompt_tokens
            if model_name:
                request["model"] = model_name
            if reasoning_effort:
                request["reasoning_effort"] = reasoning_effort

            response = requests.post(
                url=url,
                data=json.dumps(request),
                headers=self._build_headers(api_key, disable_auth_header),
                proxies=self.proxies,
                timeout=timeout,
            )
            outputs = self._parse_json_response(response, url)

            if 'error' in outputs:
                msg = self._extract_error_message(outputs)
                if "max_tokens must be at least 1" in msg:
                    raise RuntimeError(
                        f"vLLM rejected request due to context overflow: {msg}. "
                        "Try reducing data --max_seq_length, reducing output tokens, "
                        "or set --truncate_prompt_tokens (e.g., 32768 or your server max context)."
                    )
                raise RuntimeError(outputs['error'])

            return outputs['choices'][0]['text']

        request = {
            "prompt": prompts[0],
            "max_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop": stop,
        }
        # TODO: random seed is not supported?
        outputs = self._send_request(request)
        outputs = outputs['text']
        return outputs


class SGLClient(Client):
    def _single_call(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        random_seed,
        stop: List[str],
    ):
        request = {
            "text": prompts[0],
            "sampling_params": {
                "max_new_tokens": tokens_to_generate,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "stop": stop,
            }
        }
        # TODO: random seed is not supported?
        outputs = self._send_request(request)
        outputs = outputs['text']
        return outputs


class OpenAIClient:
    def __init__(
        self,
        model_name,
        **generation_kwargs
    ):  
        model2length = {
            # OpenAI
            'gpt-4': 8192,
            'gpt-4-0613': 8192,
            'gpt-4-1106-preview': 128000,
            'gpt-4-0125-preview': 128000,
            'gpt-4-turbo-preview': 128000,
            'gpt-3.5-turbo-0125': 16385,
            'gpt-3.5-turbo-1106': 16385,
            'gpt-3.5-turbo-0613': 4096,
            'gpt-3.5-turbo': 16385,
            'gpt-3.5-turbo-16k': 16385,
            'gpt-3.5-turbo-16k-0613': 16385,

            # Azure
            'gpt-4-32k': 32768,
            'gpt-4': 128000,
            'gpt-35-turbo-16k': 16384,
        }
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.azure_api_id = os.environ["AZURE_API_ID"]
        self.azure_api_secret = os.environ["AZURE_API_SECRET"]
        self.azure_api_endpoint = os.environ["AZURE_API_ENDPOINT"]
        self.model_name = model_name    
            
        # Azure
        if self.azure_api_id and self.azure_api_secret:
            if 'gpt-3.5' in model_name: self.model_name = 'gpt-35-turbo-16k'
            if 'gpt-4' in model_name: self.model_name = 'gpt-4'
        
        import tiktoken
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_length = model2length[self.model_name]
        self.generation_kwargs = generation_kwargs
        self._create_client()
        
    def _create_client(self,):
        from openai import OpenAI, AzureOpenAI
        
        # OpenAI
        if self.openai_api_key:
            self.client = OpenAI(
                api_key=self.openai_api_key
            )

        # Azure
        elif self.azure_api_id and self.azure_api_secret:
            self.client = AzureOpenAI(
                api_key=self.get_azure_api_key(
                    self.azure_api_id, 
                    self.azure_api_secret,
                    self.azure_api_endpoint,
                ),
                api_version="2024-02-15-preview",
                azure_endpoint=os.path.join(self.azure_api_endpoint, "llm/v1/azure"),
            )
        
    def _count_tokens(self, messages):
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
        
    @retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(3))
    def _send_request(self, request):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=request['msgs'],
                max_tokens=request['tokens_to_generate'],
                temperature=request['temperature'],
                seed=request['random_seed'],
                top_p=request['top_p'],
                stop=request['stop'],
            )
        except Exception as e:
            print(f"Error occurred while calling OpenAI: {e}")
            if self.azure_api_id and self.azure_api_secret and e.status_code == 401:
                # token expired
                self._create_client()
            
        return response
        
    def __call__(
        self,
        prompt: str,
    ):
        # system_msg = [{"role": "system", "content": ""}]
        system_msg = []
        user_assistant_msgs = [{"role": "user", "content": prompt}]
        msgs = system_msg + user_assistant_msgs
        openai_length = self._count_tokens(msgs)
        request = self.generation_kwargs
        
        tokens_to_generate_new = self.max_length - openai_length
        if tokens_to_generate_new < request['tokens_to_generate']:
            print(f"Reduce generate tokens from {request['tokens_to_generate']} to {tokens_to_generate_new}")
            request['tokens_to_generate'] = tokens_to_generate_new
    
        request["msgs"] = msgs
        outputs = self._send_request(request)
        response = {'text': [outputs.choices[0].message.content]}
        return response

    
    def get_azure_api_key(
        self,
        p_client_id, 
        p_client_secret, 
        p_token_url, 
        p_scope="azureopenai-readwrite",
        cache_file="azure_openai_key.json"
    ):
        base_path = Path(__file__).parent
        file_path = Path.joinpath(base_path, cache_file)
     
        # Check if the token is cached
        renew = True
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                token = json.load(f)
                renew = True if time.time() > token["expires_in"] else False

        if renew:
            # Get a new token from the OAuth server
            response = requests.post(
                os.path.join(p_token_url, "oauth/api/v1/ssa/default/token"),
                data={"grant_type": "client_credentials", "client_id": p_client_id,
                        "client_secret": p_client_secret, "scope": p_scope}
            )
            response.raise_for_status()
            token = response.json()
            token["expires_in"] += time.time()
            with open(file_path, "w") as f:
                json.dump(token, f)
     
     
        authToken = token["access_token"]
        return authToken


class GeminiClient:
    def __init__(
        self,
        model_name,
        **generation_kwargs
    ):
        model2length = {
            'gemini-1.0-pro-latest': (30720, 2048),
            'gemini-1.5-pro-latest': (1048576, 8192)
        }
        
        self.model_name = model_name
        self.model = self._initialize_model()
        self.max_input_length = model2length[model_name][0]
        self.max_output_length = model2length[model_name][1]
        assert generation_kwargs['tokens_to_generate'] < self.max_output_length, \
            print(f'tokens_to_generate exceeds {self.max_output_length}')
        
        import google.generativeai as genai        
        self.config = genai.GenerationConfig(
            candidate_count=1,
            stop_sequences=generation_kwargs['stop'],
            max_output_tokens=generation_kwargs['tokens_to_generate'],
            temperature=generation_kwargs['temperature'],
            top_p=generation_kwargs['top_p'],
            top_k=generation_kwargs['top_k'],
        )

        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    @retry(wait=wait_random_exponential(min=60, max=60), stop=stop_after_attempt(3))
    def _send_request(self, request):
        try:
            response = self.model.generate_content(request['prompt'], 
                                                   generation_config=request['config'],
                                                   safety_settings=self.safety_settings)
        except Exception as e:
            traceback.print_exc()
            return None
        return response
        
    def __call__(
        self,
        prompt: str,
    ):
        assert self.model.count_tokens(prompt).total_tokens < self.max_input_length, \
            print(f'input length exceeds {self.max_input_length}')
        
        request = {
            'prompt': prompt,
            'config': self.config,
        }
        
        outputs = self._send_request(request)

        try:
            response = {'text': [outputs.candidates[0].content.parts[0].text]}
        except Exception as e:
            response = {'text': []}
            print(outputs)
            traceback.print_exc()
            
        return response

    def _initialize_model(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        return genai.GenerativeModel(self.model_name)

