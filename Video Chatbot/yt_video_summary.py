import os
import streamlit as st
import yt_dlp as youtube_dl
from yt_dlp import DownloadError
import glob
import torch
from pytube import YouTube
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm

# Set Streamlit page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer",
    layout="centered",
    initial_sidebar_state="auto",
)

# Title of the app
st.title("📹 YouTube Video Summarizer")
st.write("Enter a YouTube video URL to generate a summary of its content.")

# # Function to download YouTube video as audio
# @st.cache_data(show_spinner=False)
# def download_youtube_audio(url, output_dir="audio/",browser="chrome",cookie_file="cookies.txt"):
#     cookie_command = f"yt-dlp --cookies-from-browser {browser} --cookies {cookie_file}"
#     os.system(cookie_command)
    
#     ydl_config = {
#         "format": "bestaudio/best",
#         "postprocessors": [
#             {
#                 "key": "FFmpegExtractAudio",
#                 "preferredcodec": "mp3",
#                 "preferredquality": "192",
#             }
#         ],
#         "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
#         "verbose": False,
#         "cookiefile": cookie_file
#     }

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     try:
#         with youtube_dl.YoutubeDL(ydl_config) as ydl:
#             ydl.download([youtube_url])
#     except DownloadError:
#         with youtube_dl.YoutubeDL(ydl_config) as ydl:
#             ydl.download([youtube_url])

    
#     # with youtube_dl.YoutubeDL(ydl_config) as ydl:
#     #     ydl.download([url])

#     audio_files = glob.glob(os.path.join(output_dir, "*.mp3"))
#     return audio_files[0] if audio_files else None

@st.cache_data(show_spinner=False)
def download_youtube_audio(url, output_dir="audio/"):
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Use pytube to download the video
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()  # Only download audio
        audio_file = stream.download(output_path=output_dir)

        return audio_file
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

# Function to transcribe audio using Whisper model
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    # processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    # model.config.forced_decoder_ids = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-small"#"openai/whisper-large-v3"

   

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    return processor, model

def transcribe_audio(processor, model, audio_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # pipe = pipeline(
    # "automatic-speech-recognition",
    # model=model,
    # tokenizer=processor.tokenizer,
    # feature_extractor=processor.feature_extractor,
    # torch_dtype=torch_dtype,
    # device=device,
    # )

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
    
    # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    # sample = dataset[0]["audio"]
    
    result = pipe(audio_path, generate_kwargs={"language": "english"})
    transcription = result["text"]
    

    
    # import soundfile as sf

    # # Load audio
    # audio_input, sampling_rate = sf.read(audio_path)
    
    # # If stereo, convert to mono by averaging channels
    # if len(audio_input.shape) > 1:
    #     audio_input = audio_input.mean(axis=1)
    
    # input_features = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_features

    # # Generate token ids
    # predicted_ids = model.generate(input_features)

    # # Decode token ids to text
    # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
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

# Function to generate summary
def generate_summary(summarizer, text):
    prompt = f"Summarize the following text:\n\n{text}"
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    summary = summarizer(prompt, **generation_args)[0]['generated_text']
    return summary

# Streamlit input for YouTube URL
youtube_url = st.text_input("YouTube Video URL", "https://www.youtube.com/watch?v=aqzxYofJ_ck")

# Button to initiate summarization
if st.button("Generate Summary"):
    if youtube_url:
        with st.spinner("Starting the summarization process..."):
            # Step 1: Download Video
            st.info("📥 Downloading YouTube video...")
            audio_file = download_youtube_audio(youtube_url)
            if audio_file:
                st.success(f"✅ Downloaded audio: {audio_file}")
            else:
                st.error("❌ Failed to download the video.")
                st.stop()

            # Step 2: Transcribe Audio
            st.info("📝 Transcribing audio...")
            processor, whisper_model = load_whisper_model()
            try:
                transcription = transcribe_audio(processor, whisper_model, audio_file)
                st.success("✅ Transcription completed!")
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")
                st.stop()

            # Display transcription (optional)
            with st.expander("View Transcription"):
                st.write(transcription)

            # Step 3: Generate Summary
            st.info("✍️ Generating summary...")
            summarizer = load_summarization_model()
            try:
                summary = generate_summary(summarizer, transcription)
                st.success("✅ Summary generated!")
            except Exception as e:
                st.error(f"❌ Summary generation failed: {e}")
                st.stop()

            # Display summary
            st.subheader("📄 Summary")
            st.write(summary)
    else:
        st.error("⚠️ Please enter a valid YouTube URL.")
