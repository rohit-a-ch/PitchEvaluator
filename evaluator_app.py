from dotenv import load_dotenv
import streamlit as st
import logging
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import threading
import librosa
from audio_recorder_streamlit import audio_recorder
import noisereduce
import torch
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import google.generativeai as genai
import io
import os
import time
import pyaudio
import wave
import sounddevice as sd
import numpy as np
from datetime import datetime
import nltk 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter

load_dotenv()
# Configure API client with your API key
genai.configure(api_key=os.getenv("API_KEY"))
emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
emojis = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😃",
    "sad": "😔",
    "angry": "😠",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😯"
}
# Define a list of common filler words
filler_words = ['um', 'uh', 'well', 'like', 'you know', 'so', 'basically', 'actually', 'right', 'I mean']
# Increase TensorFlow logging level
tf.get_logger().setLevel(logging.ERROR)

# Global variables for audio recording
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def transcribe_audio(audio):
    model="facebook/wav2vec2-large-960h-lv60-self"
    # Load the tokenizer/feature extractor
    tokenizer=Wav2Vec2Tokenizer.from_pretrained(model)
    # Load the speech-to-text model
    model=Wav2Vec2ForCTC.from_pretrained(model)

    # Perform speech-to-text
    input_values = tokenizer(audio, return_tensors="pt").input_values
    # Forward pass
    with torch.no_grad():
            logits = model(input_values).logits
    predicted_ids=torch.argmax(logits,dim=-1)

    # Decode the predicted ids token into text
    transcription=tokenizer.batch_decode(predicted_ids)[0]
    return transcription

def get_audio_duration(audio_bytes):
    RATE = 44100
    audio_bytes.seek(0)
    raw_audio = audio_bytes.read()
    audio, sr = librosa.load(io.BytesIO(raw_audio), sr=16000)
    duration = librosa.get_duration(y=audio, sr=sr)
    return audio,duration, sr

def evaluate_elevator_pitch(emotions, extracted_text, duration, username):
    # Tokenize the text
    tokenizer=TreebankWordTokenizer()
    tokens = tokenizer.tokenize(extracted_text)

    # Count the occurence of filler words
    filler_word_counts = Counter(token.lower() for token in tokens if token.lower() in filler_words)

    # Load the generative model
    model = genai.GenerativeModel('models/gemini-pro')
    prompt = f"You are an expert evaluator, tasked with assessing a pitch delivered by me [Name:{username}] for the {st.session_state.scenario_title} scenario.  Your role is to strictly act as an expert evaluator and provide feedback on various aspects of my pitch.\n\nMy goal is to capture attention and leave a lasting impression. Based on my speech for that particular {st.session_state.scenario_title} scenario, consider required aspects such as clarity, conciseness, and persuasiveness.\n\nYour evaluation should include the following each with proper scores, feedback with detail and more examples with terms/words/sentence that is used or what can be used/removed/modified to my pitch:\n\n1. Check Introduction (If anything to be added/removed/modified to my pitch for that particular scenario {st.session_state.scenario_title})\n\n 1.1 Initial Greeting: Strictly evaluate if I provided a proper initial greeting whatever required for that scenario '{st.session_state.scenario}'.\n\n 1.2 About Yourself: (If required) Evaluate if I provided my name, current role,background, experience, skills, and accomplishments. \n\n 1.3 Transition to pitch: Strictly assess the smoothness of the transition from the initial greeting to my main pitch.\n\n2. Content/Message\n\n 2.1 Problem & Need: Strictly assess does it clearly identify a specific problem?,does it establish the urgency or importance of addressing this problem? \n\n 2.2 Explained the benefit/payoff: Strictly determine if I effectively explained the benefit or payoff of my project.\n\n 2.3 Relayed relevant features: Strictly evaluate if I relayed relevant features.\n\n3. Close\n\n 3.1 Asked for the next step: Assess if I asked for the next step, such as a call or interview.\n\n 3.2 Clear Ending Evaluate if I provided a clear ending to my pitch that is expected for that particular scenario '{st.session_state.scenario_title}'.\n\n4. Communication: \n\n 4.1 Verbiage: Assess the choice of words and whether they were concise and professional if requried for that particular scenario '{st.session_state.scenario_title}'.\n\n 4.2 Time: Assess if I kept within the time limit specified (between 60 and 90 seconds).\n\n4.3 Filler Words: Asses if I used these {filler_word_counts} for many times.\n\n Emotions: Assess from the predicted emotions if I kept with the right emotions for the scenario.\n\nRelevancy: Strictly assess the context of the pitch provided by me is relevant to the exact required context of the scenario which is '[{st.session_state.scenario}]'. \n\nEvaluation Instructions:\nStrictly provide scores for that particular scenario{st.session_state.scenario_title} on a scale of 0-10 for each item, with 10 being the best possible score and 0 representing the absence of the skill or behavior being evaluated.\nAdditionally, provide relevant examples of words or sentences and compare with those used by me.\nYour assessment of the pitch's strengths and areas for improvement(If required) or suggested improvements(If required) where I missed and that's necessary for that scenario '{st.session_state.scenario_title}' with relevant examples, will provide valuable insights for refining future communication efforts(If required).Do not use NA(Not Applicable) for scores. Dont respond as third person like ""the speaker"",respond like mentioning ""you"" and evaluate as me.\n\n\n You must give me an example of how the pitch can be be delivered with proper phrases/terms, covering all the above aspects mentioned in evaluation within specified duration."    # Construct the prompt for the elevator pitch
    prompt_with_speech = f"{prompt}\n\nScenario I selected:'{st.session_state.scenario}'.\n\n \n\nMy Pitch extracted from audio:\n{extracted_text}\n\nPredicted Emotions from audio:{emotions}\nDuration of the my pitch/speech: {duration} seconds\n\n. "
   
    # Generate response based on the prompt
    response = model.generate_content(prompt_with_speech)

    # Extract and return the generated text
    generated_text = response.parts[0].text
    return generated_text

def record_audio(duration, stop_event):
    frames = []

    def callback(indata, frames, time, status):
        frames.append(indata.copy())

    # Open stream
    with sd.InputStream(channels=CHANNELS, samplerate=RATE, callback=callback):
        info = st.empty()
        time_left = st.empty()

        # Record audio until stop event is set
        start_time = time.time()
        while not stop_event.is_set():
            elapsed_time = time.time() - start_time
            remaining_time = duration - elapsed_time
            info.info("Recording...")
            if remaining_time <= 30:
                time_left.warning(f"Time remaining: {int(remaining_time)} secs")
            else:
                time_left.text(f"Time remaining: {int(remaining_time)} secs")

            if remaining_time <= 0:
                stop_event.set()

            time.sleep(0.1)

        if stop_event.is_set():
            time_left.text("")
            info.success("Recording completed!")
            time.sleep(3)
            info.text("")

    # Convert frames to a single byte stream
    audio_data = np.concatenate(frames, axis=0)
    audio_bytes = io.BytesIO()
    with wave.open(audio_bytes, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit encoding
        wf.setframerate(RATE)
        wf.writeframes(audio_data.tobytes())

    return audio_bytes.getvalue()

def countdown_timer(duration, stop_event):
    remaining_text = st.empty()
    start_time = time.time()
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        st.session_state.remaining = duration - elapsed_time
        remaining_time = st.session_state.remaining

        if remaining_time <= 0:
            st.error("Time's Up")
            stop_event.set()
            break
        elif remaining_time <= 30:
            st.warning(f"Time remaining: {int(remaining_time)} secs")
            time.sleep(1)
        else:
            remaining_text.text(f"Time remaining: {int(remaining_time)} secs")
            time.sleep(1)

def classify_audio_segments(audio, sr, segment_duration=15):
    model = load_model("model/ser_cnn_model.h5")
    # Calculate number of segments
    segment_length = segment_duration * sr
    num_segments = len(audio) // segment_length

    predictions = []

    # Iterate over audio segments
    for i in range(num_segments):
        x_predict = []
        # Extract segment
        start = i * segment_length
        end = start + segment_length
        segment = audio[start:end]

        # Extract MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40, n_fft=2048).T, axis=0)
        x_predict.append(mfccs)
        # Predict emotion label for segment

        try:
            prediction = model.predict(np.array(x_predict))
            predicted_label = emotions[np.argmax(prediction)]
            predictions.append(predicted_label)
        except Exception as e:
            st.error(f"Error occurred while classifying audio segment: {e}")
            break

    return predictions

def main():
    audio_data=None
    st.set_page_config(layout="wide")
    if 'name_submitted' not in st.session_state:
        with st.container():
            st.title("Welcome to Pitch Evaluator App")
            st.subheader("Please enter your name:")
            st.session_state.name=st.text_input("Name")
            submit_btn=st.button("Submit", key="welcome_submit",use_container_width=True)
            if submit_btn or st.session_state.name:
                st.session_state.name_submitted=True
                st.experimental_rerun() # Rerun to display the main
    else:
        username=st.session_state.name
        with st.sidebar:
            leave=st.button("Leave")
            if leave:
                del st.session_state['name_submitted']
                del st.session_state['name']
                st.experimental_rerun()  # Rerun to display the first page
            # Add a placeholder at the top of the sidebar
            sidebar_placeholder = st.sidebar.empty()
            scenarios={"Scenario 1": {"title":"Networking Event","scenario":"You're at a conference or industry gathering and bump into someone interesting. You have a limited window to make a strong impression and spark a conversation."},
                    "Scenario 2": {"title":"Coffee Chat","scenario":"You've secured a brief meeting with someone you admire or want to work with.  Use this time to showcase your value proposition and leave them wanting more."},
    "Scenario 3": {"title":"Impromptu Meeting","scenario":"You run into a potential client or collaborator in an unexpected setting.  This could be at the gym, on a plane, or even in line for coffee. A quick, captivating pitch can turn a chance encounter into an opportunity."},
    "Scenario 4": {"title":"Hallway Pitch","scenario":"You see a busy decision-maker walking down the hall. You have a fleeting moment to grab their attention and pique their interest in your idea."},
    "Scenario 5": {"title":"One-Minute Pitch Competition","scenario":"These events are becoming increasingly popular, giving entrepreneurs and creatives a platform to present their ideas in a short, fast-paced format."},
    "Scenario 6": {"title":"Social Gathering","scenario":"You're at a social gathering where you find yourself in a group conversation with potential collaborators or investors. Can you deliver a compelling pitch amidst the casual atmosphere?"},
    "Scenario 7": {"title":"Elevator Ride","scenario":"You step into an elevator with a key decision-maker in your industry. Can you pitch your idea before they reach their floor?"},
    "Scenario 8": {"title":"Job Interview","scenario":"During a job interview, the interviewer asks you to introduce yourself and your professional background. Can you craft a succinct yet impactful elevator pitch tailored to the position?"},
    "Scenario 9": {"title":"Business Lunch","scenario":"You're invited to a business lunch with potential investors. Can you deliver a persuasive pitch while maintaining a casual and professional demeanor?"},
    "Scenario 10": {"title":"Family Gathering","scenario":"During a family gathering, a relative asks you about your latest venture. Can you explain your business idea in a way that everyone, regardless of their background, can understand and find intriguing?"},
    "Scenario 11": {"title":"Sales Pitch","scenario":"You're presenting a new product to a potential client. How would you effectively communicate its features, benefits, and value proposition within the time limit"},
    "Scenario 12": {"title":"Product Launch","scenario":"You're the CEO of a company launching a product. How would you present the product at a major tech conference to capture the audience's imagination and generate buzz? You have 60-90 seconds to introduce your latest product. What sets it apart from competitors and why should they choose your product?"},
    "Scenario 13": {"title":"Expert Commentary Pitch","scenario":"You're invited by a major news network for a brief commentary on a news segment related to your field of expertise. Can you deliver insightful remarks within 60-90 seconds that capture the essence of the topic?"}
    }

            for s in scenarios.keys():
                with st.expander(f"{s}: {scenarios[s]['title']}"):
                    st.write(f"{scenarios[s]['scenario']}")
                    is_selected = st.button(f"Choose {scenarios[s]['title']}", key=f"{scenarios[s]['title']}_button")
                    if is_selected:
                        st.session_state.scenario_selected=True
                        st.session_state.scenario_title=scenarios[s]['title']
                        st.session_state.scenario=scenarios[s]['scenario']
                        selected_scenario=scenarios[s]['title']
                        audio_data=None
                        st.session_state.audio_recorded = False  # Reset flag for audio recording
                        #sidebar_placeholder.success(selected_scenario)
                        st.success("Selected")
                        
        # Create two columns 
        pitch_evaluation, record_speech = st.columns([2,1],gap="small")

        # Add content to the first column
        with pitch_evaluation:
            st.title("Pitch Evaluator")
            st.subheader(f"Hey, {username}!")
            if "scenario_selected" in st.session_state and st.session_state.scenario_selected:
                st.header(f"Selected Scenario: {st.session_state.scenario_title}")   
                st.write(st.session_state.scenario,font="Arial 1000") 
        # Start recording audio in a separate thread
            #os.remove(tmp_file)
        with record_speech:
            if "scenario_selected" in st.session_state and st.session_state.scenario_selected:
                st.text("")
                st.header("Record your pitch")
                #st.session_state.duration = st.slider("Set Recording Duration (seconds)", min_value=10, max_value=90, value=60)
                audio_data = audio_recorder(
                                            text="Click To Start / Stop Recording",
                                            sample_rate=44_100,  # Text displayed on the button (default: "Click to record")
                                            recording_color="#e8b62c",  # Color for the recording state (default: "#ff0000")
                                            neutral_color="#6aa36f",  # Color for the neutral state (default: "#d3d3d3")  # Font Awesome icon name for the button (default: "microphone")
                                            icon_size="2x",  # Size of the button icon (default: "3x")
                                            pause_threshold=5
                                        )
                if audio_data!=None:
                        st.session_state.audio_recorded=True
                        st.session_state.recorded_audio =  io.BytesIO(audio_data)
                        #print("Recorded",st.session_state.recorded_audio)
                    
        if "scenario_selected" in st.session_state and st.session_state.scenario_selected:
            # Wait for the specified duration and then stop recording
                if "audio_recorded" in st.session_state and st.session_state.audio_recorded:
                    st.session_state.audio,st.session_state.duration,st.session_state.sr = get_audio_duration(st.session_state.recorded_audio)
                    st.subheader("Your Pitch audio")
                    st.audio( st.session_state.audio, format="audio/wav",sample_rate=st.session_state.sr)
                    
                    st.write(f"Duration: {round(st.session_state.duration,2)} seconds")
                    
                    evaluate_btn=st.button("Evaluate", key="evaluate_button",use_container_width=True)
                    if evaluate_btn:
                            if "audio" in st.session_state and st.session_state.audio.any():
                                with st.spinner("Analyzing your pitch...."):
                                    predicted_emotions=classify_audio_segments(st.session_state.audio,st.session_state.sr)
                                    extracted_text = transcribe_audio(st.session_state.audio) 
                                    emotions_str="** **".join(predicted_emotions)  
                                    # Generate elevator pitch evaluation
                                    evaluation_result = evaluate_elevator_pitch(predicted_emotions,extracted_text, st.session_state.duration,st.session_state.name)
                                st.write(f"Speech: {extracted_text}")
                                st.write(f"Emotions: **{emotions_str}**")  

                                #print(extracted_text)
                                st.subheader("Pitch Evaluation:")
                                st.write(evaluation_result)
                                e,f=st.columns([1,2])
                                with e:
                                    with st.container(border=True):
                                        st.subheader("Emotions Detected")
                                        for i,emotion in enumerate(predicted_emotions):
                                            col1, col2 = st.columns([1, 1],gap="small")
                                            with col1:
                                                    st.image(f"emotions/{emotion}.png", emotion.capitalize(), width=80)
                                            with col2:
                                                    st.write(f" {i*15}-{(i+1)*15} seconds")
                                with f:
                                    filler_words_count=Counter(token.lower() for token in word_tokenize(extracted_text) if token.lower() in filler_words)
                                    filler_words_chart=go.Figure(go.Bar(x=list(filler_words_count.keys()),y=list(filler_words_count.values())),layout=dict(barcornerradius=15),)
                                    filler_words_chart.update_layout(title='Filler Words Count',xaxis_title='Filler Words',yaxis_title='Count',xaxis=dict(showline=False))
                                    st.plotly_chart(filler_words_chart,use_container_width=True)
                            else:
                                st.warning("You must record the audio to evaluate!")
                      
        else:
            st.header("Select any scenario from the left pane")
        
    
if __name__ == "__main__":
    main()
