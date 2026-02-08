from openai import OpenAI

class Base_Mining:
    def __init__(self, name: str, model: str, api_key: str, api_base: str):
        self.name = name
        self.model_name = model
        self.api_key = api_key
        self.api_base = api_base or ""


    def _build_prompt(self, *args, **kwargs) -> str:
        pass

    def _call_model(self, messages: list):
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        response = client.chat.completions.create(
            model=self.model_name,
            timeout = 800,
            messages= messages,
        )
        return response

    def generate(self, *args, **kwargs):
        prompt = self._build_prompt(*args, **kwargs)
        return self._call_model(prompt)
