import random
import openai
import time
time1 = time.time()
from openai import OpenAI
import random
import time

client = OpenAI(
    api_key="sk-123",
    base_url="***:31783/v1",
)

def llm_solve_problem(input_context):
    random.seed(10)
    start_time = time.time()
    if len(input_context)>= 2300:
        response = 'too large prompt'
        time_consume = time.time() - start_time
        return response, time_consume
    else:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_context,
                }
            ],
            model=" ",
        )
        response = chat_completion.choices[0].message.content
        time_consume = time.time() - start_time
    print(response)
    return response, time_consume

