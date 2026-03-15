from openai import OpenAI
import time
client = OpenAI(api_key="sk-123", base_url="***:31782/v1")

def llm_solve_problem(url):
    stream = False
    start_time = time.time()
    output = client.chat.completions.create(
        model=" ",  
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Summarize the content of the figure",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url,
                        },
                    },
                ],
            }
        ],
        stream=stream,
    )
    time_consume = time.time() - start_time
    return output.choices[0].message.content, time_consume
