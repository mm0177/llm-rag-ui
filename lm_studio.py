import requests
import json


def call_lm(original_prompt, selected_prompt):
    # Endpoint URL
    url = "http://localhost:1234/v1/chat/completions"

    # Headers
    headers = {
        "Content-Type": "application/json"
    }

    # Data payload
    data = {
        "messages": [
            {"role": "system", "content": "This app is to generate prompt for image generation. the user will provide Original Prompt for image generation. Based on Selected prompt, Only slightly revise Original Prompt. \
                    Please keep the Generated Prompt clear, complete, and less than 50 words. "},
            {"role": "user", "content": f"""Original Prompt: {original_prompt}\n\n
                    Selected Prompt: {selected_prompt}\n\n
                    Generated Prompt: """}
        ],
        "temperature": 0.7,
        "max_tokens": -1,   
        "stream": False
    }

    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 200:
        print("Success:")
        data = response.json()
        message = data['choices'][0]['message']['content']
        return message
    else:
        print("Error:")
        return response.text