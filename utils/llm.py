import requests
import time
import numpy as np

class GeminiAPI:
    def __init__(self, model_name, key):
        self.model_name = model_name
        self.key = key
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.key}"

    def get_model_options(self, temperature=0, per_example_max_decode_steps=150, n_sample=1, per_example_top_p=1.0):
        return {
            "temperature": temperature,
            "per_example_max_decode_steps": per_example_max_decode_steps,
            "n_sample": n_sample,
            "per_example_top_p": per_example_top_p
        }

    def generate_with_score(self, prompt, options=None, end_str=None):
        if options is None:
            options = self.get_model_options()

        # Prepare messages in the format expected by the Gemini API
        messages = [
            {
                "author": "system",
                "content": "I will give you some examples, you need to follow the examples and complete the text, and no other content.",
            },
            {"author": "user", "content": prompt},
        ]

        data = {
            "messages": messages,
            "temperature": options["temperature"],
            "candidate_count": options["n_sample"],
            "per_example_top_p": options["per_example_top_p"],
        }

        # Adjust max_decode_steps if the API supports it (check API docs)
        if options.get("per_example_max_decode_steps"):
            data["max_output_tokens"] = options["per_example_max_decode_steps"]

        # Add stop sequences if supported
        if end_str:
            data["stop_sequences"] = [end_str]

        headers = {
            'Content-Type': 'application/json',
        }

        response_json = None
        retry_num = 0
        retry_limit = 2
        error = None
        while response_json is None:
            try:
                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()
                error = None
            except requests.exceptions.RequestException as e:
                print(str(e), flush=True)
                error = str(e)
                retry_num += 1
                if retry_num > retry_limit:
                    error = "Too many retry attempts"
                    # Handle the error by returning a placeholder response
                    response_json = {
                        "candidates": [{"content": "PLACEHOLDER"}]
                    }
                    break
                else:
                    time.sleep(60)

        if error:
            raise Exception(error)

        if "candidates" not in response_json:
            return [("No content returned", 0)]

        results = []
        for i, res in enumerate(response_json["candidates"]):
            text = res["content"]
            # Simulate a confidence score
            fake_conf = (len(response_json["candidates"]) - i) / len(response_json["candidates"])
            results.append((text, np.log(fake_conf)))

        return results


    def generate(self, prompt, options=None, end_str=None):
        if options is None:
            options = self.get_model_options()
        options["n_sample"] = 1
        result = self.generate_plus_with_score(prompt, options, end_str)[0][0]
        return result
