import streamlit as st
from google import genai
from google.genai import types
import os
import random
import time
from google.oauth2 import service_account

# --- CONFIGURATION ---

gcp_info = st.secrets["gcp_service_account"]
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
credentials = service_account.Credentials.from_service_account_info(
    gcp_info, 
    scopes=SCOPES
)
LOCATION = "us-central1"
MODEL_ID = "gemini-2.5-flash" # Current stable version

client = genai.Client(
    vertexai=True, 
    project=gcp_info["project_id"], 
    location=LOCATION,
    credentials=credentials 
)
# --- SESSION STATE FOR BUTTON LOCKING ---
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "page" not in st.session_state:
    st.session_state.page = "Custom Files"

if "question_type" not in st.session_state:
    st.session_state.question_type = "True or False"

if "quantity" not in st.session_state:
    st.session_state.quantity = 1

if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = None

if "user_answers" not in st.session_state:
    st.session_state.user_answers = []

if "feedback_response" not in st.session_state:
    st.session_state.feedback_response = None

if "followup_prompt" not in st.session_state:
    st.session_state.followup_prompt = None

if "followup_response" not in st.session_state:
    st.session_state.followup_response = None

if "hkdse_followup_response" not in st.session_state:
    st.session_state.hkdse_followup_response = None

if "continue_conversation" not in st.session_state:
    st.session_state.continue_conversation = ""

if "answers_submitted" not in st.session_state:
    st.session_state.answers_submitted = False

if "followup_conversation" not in st.session_state:
    st.session_state.followup_conversation = ""

if "hkdse_question_type" not in st.session_state:
    st.session_state.hkdse_question_type = "True or False"

if "hkdse_quantity" not in st.session_state:
    st.session_state.hkdse_quantity = 1

if "hkdse_category" not in st.session_state:
    st.session_state.hkdse_category = "Chinese"

if "hkdse_generated_questions" not in st.session_state:
    st.session_state.hkdse_generated_questions = None

if "hkdse_user_answers" not in st.session_state:
    st.session_state.hkdse_user_answers = []

if "hkdse_feedback_response" not in st.session_state:
    st.session_state.hkdse_feedback_response = None

if "hkdse_answers_submitted" not in st.session_state:
    st.session_state.hkdse_answers_submitted = False

if "hkdse_followup_conversation" not in st.session_state:
    st.session_state.hkdse_followup_conversation = ""

def start_processing():
    st.session_state.is_processing = True

st.set_page_config(page_title="AI Study Tutor", page_icon="🎓")

st.title("🎓 AI Study Tutor")

# Sidebar for page selection
if st.sidebar.button("Custom Files"):
    st.session_state.page = "Custom Files"
    st.session_state.generated_questions = None
    st.session_state.feedback_response = None
    st.session_state.followup_response = None
    st.session_state.continue_conversation = ""
    st.session_state.user_answers = []
    st.session_state.answers_submitted = False
    st.session_state.followup_conversation = ""

if st.sidebar.button("HK DSE"):
    st.session_state.page = "HK DSE"
    st.session_state.generated_questions = None
    st.session_state.feedback_response = None
    st.session_state.followup_response = None
    st.session_state.continue_conversation = ""
    st.session_state.user_answers = []
    st.session_state.answers_submitted = False
    st.session_state.followup_conversation = ""

if st.session_state.page == "Custom Files":
    st.markdown("Upload a document, and I'll generate custom study questions for you.")

    # Question type selection
    question_type = st.radio("Select Question Type", ["True or False", "Multiple Choice", "Short Answer"], key="question_type_radio")

    if question_type in ["True or False", "Multiple Choice"]:
        quantity = st.selectbox("Number of Questions", [1, 3, 5], key="quantity_select")
    else:
        quantity = st.selectbox("Number of Questions", [1, 2, 3], key="quantity_select")

    st.session_state.question_type = question_type
    st.session_state.quantity = quantity

    # File Uploader
    uploaded_files = st.file_uploader("Upload your study notes (PDF)", type=["pdf"], accept_multiple_files=True, max_upload_size=10)

    if uploaded_files:
        if st.button(
            "Generate Questions", 
        ):
            with st.spinner("Analyzing document..."):
                try:
                    # Create PDF parts for all uploaded files
                    pdf_parts = []
                    for uploaded_file in uploaded_files:
                        file_bytes = uploaded_file.getvalue()
                        pdf_part = types.Part.from_bytes(
                            data=file_bytes,
                            mime_type="application/pdf"
                        )
                        pdf_parts.append(pdf_part)
                    
                    if st.session_state.question_type == "Multiple Choice":
                        prompt = f"You are a professional tutor. Based on this document, generate {st.session_state.quantity} multiple choice study questions with hints. Each question should have 4 answer options labeled a, b, c, d. Do not provide any answers. Instead of using ** for bold, or * for italics, only use capitals for emphasis"
                    elif st.session_state.question_type == "True or False":
                        prompt = f"You are a professional tutor. Based on this document, generate {st.session_state.quantity} true or false study questions with hints. Do not provide any answers. Instead of using ** for bold, or * for italics, only use capitals for emphasis"
                    else:
                        prompt = f"You are a professional tutor. Based on this document, generate {st.session_state.quantity} {st.session_state.question_type.lower()} study questions with hints. Do not provide any answers. Instead of using ** for bold, or * for italics, only use capitals for emphasis"

                    # COMMENT THIS BACK IN WHEN YOU WANT TO TEST WITH THE MODEL
                    response = client.models.generate_content(
                        model=MODEL_ID,
                        contents=[pdf_part, prompt]
                    ).text

                    # COMMENT THIS BACK IN WHEN YOU DON'T WANT TO TEST WITH THE MODEL
                    # time.sleep(1)
                    # response = random.randint(0,10)

                    st.success("Done!")
                    st.session_state.generated_questions = str(response)
                    st.session_state.feedback_response = None
                    st.session_state.followup_response = None
                    st.session_state.continue_conversation = ""
                    st.session_state.user_answers = []
                    st.session_state.answers_submitted = False
                    st.session_state.followup_conversation = ""
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    
    # Display generated questions persistently
    if st.session_state.generated_questions:
        st.markdown("---")
        st.markdown("### Questions")
        st.text(st.session_state.generated_questions)

    # Answer input section
    if st.session_state.generated_questions:
        st.markdown("### Provide Your Answers")
        answers = []
        for i in range(st.session_state.quantity):
            if st.session_state.question_type == "True or False":
                answer = st.radio(f"Answer for Question {i+1}", ["T", "F"], key=f"answer_{i}")
            elif st.session_state.question_type == "Multiple Choice":
                answer = st.radio(f"Answer for Question {i+1}", ["A", "B", "C", "D"], key=f"answer_{i}")
            else:  # Short Answer
                answer = st.text_input(f"Answer for Question {i+1}", key=f"answer_{i}")
            answers.append(answer)
        
        if st.button("Submit Answers", disabled=st.session_state.answers_submitted):
            with st.spinner("Checking answers..."):
                try:
                    feedback_prompt = f"Here are the questions: {st.session_state.generated_questions}\n\nUser's answers: {', '.join(answers)}\n\nAs a tutor, provide detailed feedback on whether each answer is correct, explain why, and suggest improvements if needed."
                    
                    # COMMENT THIS BACK IN WHEN YOU WANT TO TEST WITH THE MODEL
                    feedback_response = client.models.generate_content(
                        model=MODEL_ID,
                        contents=[feedback_prompt]
                    ).text

                    # COMMENT THIS BACK IN WHEN YOU DONT WANT TO TEST WITH THE MODEL
                    # time.sleep(3)
                    # feedback_response = f"Feedback for answers: {', '.join(answers)} - This is mock feedback."
                    
                    st.session_state.user_answers = answers
                    st.session_state.feedback_response = feedback_response
                    st.session_state.followup_response = None
                    st.session_state.answers_submitted = True
                    st.session_state.followup_conversation = ""
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error getting feedback: {e}")
        
        # Show feedback and continuation if feedback exists
        if st.session_state.feedback_response:
            st.success("Feedback received!")
            st.markdown("### Feedback")
            st.markdown(st.session_state.feedback_response)
            
            with st.form("followup_form"):
                continue_text = st.text_input(
                    "Continue the conversation",
                    key="continue_conversation",
                    value="",
                )
                submit_followup = st.form_submit_button("Submit follow-up")

                if submit_followup and continue_text:
                    # Build the full conversation context
                    if not st.session_state.followup_conversation:
                        # First follow-up: include original context
                        st.session_state.followup_conversation = (
                            f"Original questions: {st.session_state.generated_questions}\n"
                            f"User answers: {', '.join(st.session_state.user_answers)}\n"
                            f"AI feedback: {st.session_state.feedback_response}\n"
                            f"---\n"
                        )
                    
                    # Append user input to conversation
                    st.session_state.followup_conversation += f"User: {continue_text}\n"
                    
                    # Get AI response with full conversation context
                    full_prompt = (
                        f"{st.session_state.followup_conversation}"
                        f"As a tutor, continue the conversation and provide helpful guidance based on all context above."
                    )
                    
                    # COMMENT THIS BACK IN WHEN YOU WANT TO TEST WITH THE MODEL
                    response = client.models.generate_content(
                        model=MODEL_ID,
                        contents=[full_prompt]
                    )
                    ai_response = response.text

                    # COMMENT THIS BACK IN WHEN YOU DONT WANT TO TEST WITH THE MODEL
                    # ai_response = str(random.randint(0, 100))
                    
                    # Append AI response to conversation
                    st.session_state.followup_conversation += f"AI: {ai_response}\n"
                    st.session_state.followup_response = ai_response

                latest_ai = st.session_state.followup_response if st.session_state.followup_response else st.session_state.feedback_response
                if latest_ai:
                    st.markdown("### Latest AI Response")
                    st.markdown(latest_ai)

                if st.session_state.followup_conversation:
                    st.markdown("### Conversation History")
                    st.markdown(st.session_state.followup_conversation)

if st.session_state.page == "HK DSE":
        st.markdown("### HK DSE Question Generator")
        hkdse_category = st.radio(
            "Select HK DSE Category",
            ["Chinese", "English", "Math", "Physics", "Chemistry", "Biology", "Economics"],
            key="hkdse_category_radio"
        )
        hkdse_question_type = st.radio(
            "Select Question Type",
            ["True or False", "Multiple Choice", "Short Answer"],
            key="hkdse_question_type_radio"
        )

        if hkdse_question_type in ["True or False", "Multiple Choice"]:
            hkdse_quantity = st.selectbox("Number of Questions", [1, 3, 5], key="hkdse_quantity_select")
        else:
            hkdse_quantity = st.selectbox("Number of Questions", [1, 2, 3], key="hkdse_quantity_select")

        st.session_state.hkdse_category = hkdse_category
        st.session_state.hkdse_question_type = hkdse_question_type
        st.session_state.hkdse_quantity = hkdse_quantity

        # Optional file upload
        hkdse_uploaded_files = st.file_uploader("Upload study materials (optional, PDF)", type=["pdf"], accept_multiple_files=True, max_upload_size=200, key="hkdse_file_uploader")

        if st.button("Generate Questions", key="hkdse_generate"):
            with st.spinner("Generating questions..."):
                try:
                    # Create PDF parts if files are uploaded
                    pdf_parts = []
                    if hkdse_uploaded_files:
                        for uploaded_file in hkdse_uploaded_files:
                            file_bytes = uploaded_file.getvalue()
                            pdf_part = types.Part.from_bytes(
                                data=file_bytes,
                                mime_type="application/pdf"
                            )
                            pdf_parts.append(pdf_part)
                    
                    if st.session_state.hkdse_question_type == "Multiple Choice":
                        if pdf_parts:
                            prompt = f"You are a professional tutor. Based on the uploaded materials, generate {st.session_state.hkdse_quantity} multiple choice questions for HK DSE {st.session_state.hkdse_category}. Each question should have 4 answer options labeled a, b, c, d. Do not provide any answers. Instead of using ** for bold, or * for italics, only use capitals for emphasis"
                        else:
                            prompt = f"You are a professional tutor. Generate {st.session_state.hkdse_quantity} multiple choice questions for HK DSE {st.session_state.hkdse_category}. Each question should have 4 answer options labeled a, b, c, d. Do not provide any answers. Instead of using ** for bold, or * for italics, only use capitals for emphasis"
                    elif st.session_state.hkdse_question_type == "True or False":
                        if pdf_parts:
                            prompt = f"You are a professional tutor. Based on the uploaded materials, generate {st.session_state.hkdse_quantity} true or false questions for HK DSE {st.session_state.hkdse_category} with hints. Do not provide any answers. Instead of using ** for bold, or * for italics, only use capitals for emphasis"
                        else:
                            prompt = f"You are a professional tutor. Generate {st.session_state.hkdse_quantity} true or false questions for HK DSE {st.session_state.hkdse_category} with hints. Do not provide any answers. Instead of using ** for bold, or * for italics, only use capitals for emphasis"
                    else:
                        if pdf_parts:
                            prompt = f"You are a professional tutor. Based on the uploaded materials, generate {st.session_state.hkdse_quantity} short answer questions for HK DSE {st.session_state.hkdse_category} with hints. Do not provide any answers. Instead of using ** for bold, or * for italics, only use capitals for emphasis"
                        else:
                            prompt = f"You are a professional tutor. Generate {st.session_state.hkdse_quantity} short answer questions for HK DSE {st.session_state.hkdse_category} with hints. Do not provide any answers. Instead of using ** for bold, or * for italics, only use capitals for emphasis"

                    # COMMENT THIS BACK IN WHEN YOU WANT TO TEST WITH THE MODEL
                    response = client.models.generate_content(
                        model=MODEL_ID,
                        contents=pdf_parts + [prompt] if pdf_parts else [prompt]
                    ).text

                    # COMMENT THIS BACK IN WHEN YOU DONT WANT TO TEST WITH THE MODEL
                    # time.sleep(1)
                    # response = random.randint(0, 10)

                    st.success("Done!")
                    st.session_state.hkdse_generated_questions = str(response)
                    st.session_state.hkdse_feedback_response = None
                    st.session_state.hkdse_answers_submitted = False
                    st.session_state.hkdse_followup_response = None
                    st.session_state.hkdse_followup_conversation = ""
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # Display generated questions persistently
        if st.session_state.hkdse_generated_questions:
            st.markdown("---")
            st.markdown("### Questions")
            st.text(st.session_state.hkdse_generated_questions)

        # Answer input section
        if st.session_state.hkdse_generated_questions:
            st.markdown("### Provide Your Answers")
            hkdse_answers = []
            for i in range(st.session_state.hkdse_quantity):
                if st.session_state.hkdse_question_type == "True or False":
                    answer = st.radio(f"Answer for Question {i+1}", ["T", "F"], key=f"hkdse_answer_{i}")
                elif st.session_state.hkdse_question_type == "Multiple Choice":
                    answer = st.radio(f"Answer for Question {i+1}", ["A", "B", "C", "D"], key=f"hkdse_answer_{i}")
                else:  # Short Answer
                    answer = st.text_input(f"Answer for Question {i+1}", key=f"hkdse_answer_{i}")
                hkdse_answers.append(answer)
            
            if st.button("Submit Answers", disabled=st.session_state.hkdse_answers_submitted, key="hkdse_submit_answers"):
                with st.spinner("Checking answers..."):
                    try:
                        feedback_prompt = f"Here are the HK DSE {st.session_state.hkdse_category} questions: {st.session_state.hkdse_generated_questions}\n\nUser's answers: {', '.join(hkdse_answers)}\n\nAs a tutor, provide detailed feedback on whether each answer is correct, explain why, and suggest improvements if needed."
                        
                        # COMMENT THIS BACK IN WHEN YOU WANT TO TEST WITH THE MODEL
                        feedback_response = client.models.generate_content(
                            model=MODEL_ID,
                            contents=[feedback_prompt]
                        ).text

                        # COMMENT THIS BACK IN WHEN YOU DONT WANT TO TEST WITH THE MODEL
                        # time.sleep(3)
                        # feedback_response = f"Feedback for answers: {', '.join(hkdse_answers)} - This is mock feedback."
                        
                        st.session_state.hkdse_user_answers = hkdse_answers
                        st.session_state.hkdse_feedback_response = feedback_response
                        st.session_state.hkdse_answers_submitted = True
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error getting feedback: {e}")
            
            # Show feedback and continuation if feedback exists
            if st.session_state.hkdse_feedback_response:
                st.success("Feedback received!")
                st.markdown("### Feedback")
                st.markdown(st.session_state.hkdse_feedback_response)
                
                with st.form("hkdse_followup_form"):
                    continue_text = st.text_input(
                        "Continue the conversation",
                        key="hkdse_continue_conversation",
                        value="",
                    )
                    submit_followup = st.form_submit_button("Submit follow-up", key="hkdse_submit_followup")

                    if submit_followup and continue_text:
                        # Build the full conversation context
                        if not st.session_state.hkdse_followup_conversation:
                            # First follow-up: include original context
                            st.session_state.hkdse_followup_conversation = (
                                f"Original questions: {st.session_state.hkdse_generated_questions}\n"
                                f"User answers: {', '.join(st.session_state.hkdse_user_answers)}\n"
                                f"AI feedback: {st.session_state.hkdse_feedback_response}\n"
                                f"---\n"
                            )
                        
                        # Append user input to conversation
                        st.session_state.hkdse_followup_conversation += f"User: {continue_text}\n"
                        
                        # Get AI response with full conversation context
                        full_prompt = (
                            f"{st.session_state.hkdse_followup_conversation}"
                            f"As a tutor, continue the conversation and provide helpful guidance based on all context above."
                        )
                        
                        # COMMENT THIS BACK IN WHEN YOU WANT TO TEST WITH THE MODEL
                        response = client.models.generate_content(
                            model=MODEL_ID,
                            contents=[full_prompt]
                        )
                        ai_response = response.text

                        # COMMENT THIS BACK IN WHEN YOU DONT WANT TO TEST WITH THE MODEL
                        # ai_response = str(random.randint(0, 100))
                        
                        # Append AI response to conversation
                        st.session_state.hkdse_followup_conversation += f"AI: {ai_response}\n"
                        st.session_state.hkdse_followup_response = ai_response

                    latest_hkdse_ai = st.session_state.hkdse_followup_response if st.session_state.hkdse_followup_response else st.session_state.hkdse_feedback_response
                    if latest_hkdse_ai:
                        st.markdown("### Latest AI Response")
                        st.markdown(latest_hkdse_ai)

                    if st.session_state.hkdse_followup_conversation:
                        st.markdown("### Conversation History")
                        st.markdown(st.session_state.hkdse_followup_conversation)
