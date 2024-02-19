from pathlib import Path
import new
import playsound
from openai import OpenAI
import constants
import os
os.environ["OPENAI_API_KEY"] = constants.APIKEY
client = OpenAI()

speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input=new.retrive()
)

response.stream_to_file(speech_file_path)
playsound.playsound("speech.mp3")
