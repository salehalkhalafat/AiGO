from openai import OpenAI
from pathlib import Path
import constants
import os
os.environ["OPENAI_API_KEY"] = constants.APIKEY
client = OpenAI()

audio_file = Path(__file__).parent / "speech.mp3"
transcript = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file,
  response_format="text"
)
print(transcript)