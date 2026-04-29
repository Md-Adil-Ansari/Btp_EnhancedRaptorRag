import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass

    @abstractmethod
    def extract_context(self, context, max_tokens=20):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def extract_context(self, context, max_tokens=20, stop_sequence=None):
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Extract a 2-3 word context or NER tag representing the main domain or intent of the following text: {context}",
                    },
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            return ""


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def extract_context(self, context, max_tokens=20, stop_sequence=None):
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Extract a 2-3 word context or NER tag representing the main domain or intent of the following text: {context}",
                    },
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            return ""


class OllamaSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="qwen3:30b"):
        import ollama
        self.model = model
        self.client = ollama.Client()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        try:
            prompt = f"Write a summary of the following, including as many key details as possible: {context}:"
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                keep_alive=-1
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(e)
            return str(e)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def extract_context(self, context, max_tokens=20, stop_sequence=None):
        try:
            prompt = f"Extract a 2-3 word context or NER tag representing the main domain or intent of the following text: {context}"
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                keep_alive=-1
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(e)
            return ""


import time
import threading

class GemmaSummarizationModel(BaseSummarizationModel):
    _api_lock = threading.Lock()
    _last_call_time = 0.0

    def __init__(self, model="gemma-3-27b-it"):
        """
        Initializes the Gemma model with the specified model version.

        Args:
            model (str, optional): The Gemma model version to use for answering. Defaults to "gemma-2-27b-it".
        """
        from google import genai
        
        self.model = model
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set to use GemmaSummarizationModel.")
            
        self.client = genai.Client(api_key=api_key)

    def _enforce_rate_limit(self):
        with self._api_lock:
            elapsed = time.time() - GemmaSummarizationModel._last_call_time
            if elapsed < 3.0:
                time.sleep(3.0 - elapsed)
            GemmaSummarizationModel._last_call_time = time.time()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        try:
            self._enforce_rate_limit()
            prompt = (
                f"Write a summary of the following, including as many key details as possible: {context}:"
            )
            
            with open("gemma_summarization_prompts.txt", "a", encoding="utf-8") as f:
                f.write(prompt + "\n" + "="*80 + "\n\n")
                
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(e)
            return str(e)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def extract_context(self, context, max_tokens=20, stop_sequence=None):
        try:
            self._enforce_rate_limit()
            prompt = (
                f"Extract a 2-3 word context or NER tag representing the main domain or intent of the following text.\n"
                f"IMPORTANT: RETURN ONLY THE TAG. DO NOT include any conversational filler, explanations, or quotes. ONLY the 2-3 words.\n\n"
                f"Text: {context}"
            )
            
            with open("gemma_summarization_prompts.txt", "a", encoding="utf-8") as f:
                f.write("CONTEXT EXTRACTION: " + prompt + "\n" + "="*80 + "\n\n")
                
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(e)
            return ""

