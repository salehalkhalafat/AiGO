from openai import OpenAI
import constants
import os
import STT
os.environ["OPENAI_API_KEY"] = constants.APIKEY
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": STT.transcript},
  ]
)
print(response.choices[0].message.content)
