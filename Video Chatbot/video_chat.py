import os
import re
import streamlit as st
from moviepy.editor import VideoFileClip
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

# Set Streamlit page configuration
st.set_page_config(
    page_title="Video Transcription and Chat Summarizer",
    layout="centered",
    initial_sidebar_state="auto",
)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []  # Initialize messages session

# Clear the chat history
def clear_conversation():
    st.session_state['messages'] = []

# Button to clear chat history
if st.button("Clear Conversation"):
    clear_conversation()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Function to convert uploaded video to audio

# Function to convert uploaded video to audio
def convert_video_to_audio(video_path, output_audio_path="audio/uploaded_audio.wav"):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_audio_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load video and extract audio
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)
    return output_audio_path


# Function to transcribe audio using Whisper model
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-small"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    return processor, model

def transcribe_audio(processor, model, audio_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device
    )

    result = pipe(audio_path, generate_kwargs={"language": "english"})
    transcription = result["text"]
    return transcription

# Function to load summarization model
@st.cache_resource(show_spinner=False)
def load_summarization_model():
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        device_map="auto",  # Automatically selects GPU if available
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    summarizer = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return summarizer

# Function to generate summary with bullet points for key highlights
def generate_summary(summarizer, text):
    prompt = f"Summarize the following text and highlight key points with bullet points:\n\n{text}"
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    summary = summarizer(prompt, **generation_args)[0]['generated_text']
    return summary

# Video file upload
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    with st.spinner("Converting video to audio..."):
        # Save the uploaded file to a temporary location
        video_path = f"temp_video.{uploaded_file.name.split('.')[-1]}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Convert video to audio
        audio_file = convert_video_to_audio(video_path)
        st.success(f"✅ Converted video to audio: {audio_file}")

        # Transcribe the audio
        st.info("📝 Transcribing audio...")
        processor, whisper_model = load_whisper_model()
        transcription = transcribe_audio(processor, whisper_model, audio_file)
        st.success("✅ Transcription completed!")

        # Display transcription (optional)
        with st.expander("View Transcription"):
            st.write(transcription)

        # Generate summary with key points
        st.info("✍️ Generating summary...")
        summarizer = load_summarization_model()
        summary = generate_summary(summarizer, transcription)
        st.success("✅ Summary generated!")

        # Display summary
        st.subheader("📄 Summary")
        st.write(summary)

        # Add first summary to chat history
        if not st.session_state['messages']:
            st.session_state.messages.append({"role": "assistant", "content": summary})

# Chat interface to ask questions based on the transcription
if prompt := st.chat_input("Ask a question about the video transcription..."):
    # Display user message in chat
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate a response from the assistant
    st.info("💬 Generating response...")
    messages = st.session_state.messages

    # For chat interface, use the summarizer as a conversational model
    text_prompt = f"Based on the video transcription and summary, answer the following question: {prompt}\n\n"

    response = generate_summary(summarizer, text_prompt)
    response_cleaned = re.sub(r'<.*?>', '', response)  # Clean up any tags from the response

    # Display assistant response in chat
    st.chat_message("assistant").write(response_cleaned)
    st.session_state.messages.append({"role": "assistant", "content": response_cleaned})