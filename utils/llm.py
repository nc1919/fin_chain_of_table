import requests
import time
import numpy as np

class GeminiAPI:
    def __init__(self, model_name, key):
        self.model_name = model_name
        self.key = key
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.key}"

    def get_model_options(self, temperature=0, max_decode_steps=150, n_sample=1):
        return {
            "temperature": temperature,
            "max_decode_steps": max_decode_steps,
            "n_sample": n_sample,
        }

    def generate_with_score(self, prompt, options=None):
        if options is None:
            options = self.get_model_options()
        
        headers = {
            'Content-Type': 'application/json',
        }

        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "temperature": options["temperature"],
            "n_sample": options["n_sample"],
        }

        response = None
        retry_num = 0
        retry_limit = 2
        error = None
        while response is None:
            try:
                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()
                error = None
            except requests.exceptions.RequestException as e:
                print(str(e), flush=True)
                error = str(e)
                if retry_num > retry_limit:
                    error = "too many retry attempts"
                    return [{"text": "PLACEHOLDER"}]
                else:
                    time.sleep(60)
                retry_num += 1
        
        if error:
            raise Exception(error)
        
        response_json = response.json()
        results = []
        for i, res in enumerate(response_json.get("contents", [])):
            text = res["parts"][0]["text"]
            fake_conf = (len(response_json["contents"]) - i) / len(response_json["contents"])
            results.append((text, np.log(fake_conf)))
        
        return results

    def generate(self, prompt, options=None):
        if options is None:
            options = self.get_model_options()
        options["n_sample"] = 1
        result = self.generate_with_score(prompt, options)[0][0]
        return result
