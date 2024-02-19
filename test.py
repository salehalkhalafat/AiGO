from pathlib import Path
import record
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
from openai import OpenAI
from pydub import AudioSegment  # Import pydub
os.environ["OPENAI_API_KEY"] = constants.APIKEY
client = OpenAI()
def record():
    # Record in chunks of 1024 samples
    chunk = 1024 
    
    # 16 bits per sample
    sample_format = pyaudio.paInt16  
    chanels = 2
    
    # Record at 44400 samples per second
    smpl_rt = 44400 
    seconds = 4
    filename = "path_of_file.wav"
    
    # Create an interface to PortAudio
    pa = pyaudio.PyAudio()  
    
    stream = pa.open(format=sample_format, channels=chanels, 
                    rate=smpl_rt, input=True, 
                    frames_per_buffer=chunk)
    
    print('Recording...')
    
    # Initialize array that be used for storing frames
    frames = []  
    
    # Store data in chunks for 8 seconds
    for i in range(0, int(smpl_rt / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    
    # Terminate - PortAudio interface
    pa.terminate()
    
    print('Done !!! ')
    
    # Save the recorded data in a .wav format
    sf = wave.open(filename, 'wb')
    sf.setnchannels(chanels)
    sf.setsampwidth(pa.get_sample_size(sample_format))
    sf.setframerate(smpl_rt)
    sf.writeframes(b''.join(frames))
    sf.close()
    wave_audio = AudioSegment.from_wav(filename)
    mp3_filename = "speech.mp3"
    wave_audio.export(mp3_filename, format="mp3")


def TTS():
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=retrive()
    )
    response.stream_to_file(speech_file_path)

def STT():
    audio_file = Path(__file__).parent / "speech.mp3"
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    response_format="text"
    )
    return transcript

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
    text = STT()
    query = text
    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    return result['answer']



record()
STT()
TTS()
playsound.playsound("speech.mp3")

