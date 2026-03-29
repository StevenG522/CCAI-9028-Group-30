import streamlit as st
from google import genai
from google.genai import types
import os

# --- CONFIGURATION ---
PROJECT_ID = "project-6ca3a1f6-ee18-4141-bf7" # From your error log
LOCATION = "us-central1"
MODEL_ID = "gemini-2.5-flash" # Current stable version

# Initialize Vertex AI Client
# When deployed to Cloud Run, this automatically uses the service account attached to the container
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# --- SESSION STATE FOR BUTTON LOCKING ---
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

def start_processing():
    st.session_state.is_processing = True

st.set_page_config(page_title="AI Study Tutor", page_icon="🎓")

st.title("🎓 AI Study Tutor")
st.markdown("Upload a document, and I'll generate custom study questions for you.")

# File Uploader
uploaded_file = st.file_uploader("Upload your study notes (PDF)", type=["pdf"])

if uploaded_file:
    if st.button(
        "Generate Questions", 
        disabled=st.session_state.is_processing, 
        on_click=start_processing
    ):
        with st.spinner("Analyzing document..."):
            try:
                # Read file as bytes
                file_bytes = uploaded_file.getvalue()
                
                # Create the prompt parts
                # We send the PDF bytes directly to Vertex AI
                pdf_part = types.Part.from_bytes(
                    data=file_bytes,
                    mime_type="application/pdf"
                )
                
                prompt = "You are a professional tutor. Based on this document, generate 1 challenging study questions with hints."

                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[pdf_part, prompt]
                )

                # response = client.models.generate_content(
                #     model=MODEL_ID,
                #     contents=["say any 5 letter word"]
                # )

                st.success("Done!")
                st.markdown("---")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

            finally:
                # 5. Unlock the button for the next run
                st.session_state.is_processing = False

st.sidebar.info("Deploying to Cloud Run ensures secure, keyless authentication.")
