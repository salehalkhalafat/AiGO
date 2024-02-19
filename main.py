from pathlib import Path
import record
import RPi.GPIO as GPIO
import playsound
import os
import sys
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, WebBaseLoader, TextLoader, PyPDFLoader, GitLoader, CSVLoader, PythonLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains.summarize import load_summarize_chain, _load_stuff_chain
import constants
import pyaudio
import wave
from pydub import audio_segment
import time
from pydub import AudioSegment  # Import pydub


os.environ["OPENAI_API_KEY"] = constants.APIKEY
client = OpenAI()
def button_pressed(channel):
    global start_time
    start_time = time.time()


# Callback function to stop recording when the button is released
def button_released(channel):
    duration = time.time() - start_time
    record(duration)

def record(duration):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2  # Stereo
    sample_rate = 44100  # Record at 44100 samples per second
    filename = "path_of_file.wav"

    # Create PyAudio object
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=sample_rate,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    print("Recording...")

    # Record audio for the specified duration
    start_time = time.time()
    while time.time() - start_time < duration:
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the PyAudio object
    p.terminate()

    print("Finished recording.")

    # Save the recorded audio as a WAV file
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()

    # Convert WAV to MP3
    wave_audio = AudioSegment.from_wav(filename)
    mp3_filename = "speech.mp3"
    wave_audio.export(mp3_filename, format="mp3")



def TTS():
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=new.retrive()
    )
    response.stream_to_file(speech_file_path)

def STT():
    audio_file = Path(__file__).parent / "speech.mp3"
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    response_format="text"
    )

def retrive():
    # Enable to save to disk & reuse the model (for repeated queries on the same data)
    PERSIST = False
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
    query = None
    if len(sys.argv) > 1:
        query = sys.argv[1]

    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        # loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
        loader = DirectoryLoader("Data/")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    text = STT.transcript
    query = text
    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    return result['answer']
    
def main():
    # Set up GPIO mode
    GPIO.setmode(GPIO.BCM)

    # Set up GPIO pin 17 as input with pull-up resistor
    GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Add event detection for button press and release
    GPIO.add_event_detect(4, GPIO.FALLING, callback=button_pressed, bouncetime=300)
    GPIO.add_event_detect(4, GPIO.RISING, callback=button_released, bouncetime=300)
    STT()
    retrive()
    TTS()
    try:
        print("Press and hold the button to record.")
        while True:
            pass  # Keep the script running

    except KeyboardInterrupt:
        # Clean up GPIO settings
        GPIO.cleanup()


if __name__ == "__main__":
    main()
