import streamlit as st
from google import genai
from google.genai import types
import os
import random
import time
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview import rag

# --- CONFIGURATION ---

gcp_info = st.secrets["gcp_service_account"]
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
credentials = service_account.Credentials.from_service_account_info(
    gcp_info, 
    scopes=SCOPES
)
LOCATION = "asia-northeast1"
MODEL_ID = "gemini-2.5-flash" # Current stable version
PROJECT_ID = gcp_info["project_id"]
client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location=LOCATION,
    credentials=credentials 
)


CORPUS_DISPLAY_NAME = "HKDSE_PDFs"
def get_or_create_corpus():
    corpora = list(rag.list_corpora())
    for c in corpora:
        if c.display_name == CORPUS_DISPLAY_NAME:
            return c
    return rag.create_corpus(display_name=CORPUS_DISPLAY_NAME)
my_corpus = get_or_create_corpus()

vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=my_corpus.name,
                )
            ],
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=10,
                filter=rag.utils.resources.Filter(vector_distance_threshold=0.5),
            ),
        ),
    )
)

rag_model = GenerativeModel(
    model_name="gemini-2.5-flash", tools=[rag_retrieval_tool]
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

if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = ""

if "hkdse_custom_prompt" not in st.session_state:
    st.session_state.hkdse_custom_prompt = ""

def start_processing():
    st.session_state.is_processing = True

def get_prompt_formatting_string():
    return "Please format the multiple-choice questions using the following structure:\
    Use a header (##) for the question title.\
    Use a numbered list (1., 2.) for the questions.\
    Never generate LATEX text as it will not render properly    \
    Use an uppercase lettered list (A., B., C., D.) for the options, ensuring each option is on a new line.\
    Use a blockquote (>) for the marking scheme. NEVER generate a marking scheme that reveals the correct or incorrect answers when creating questions.\
    Use bold text for key terms like CORRECT or LENGTH."


def get_tutor_prompt():
    return f"""Role: You are an expert tutor specializing in question generation. You have 10+ years of experience generating thought-provoking questions in various subjects.
    Background: I am a secondary school student preparing for my exams.
    Constraints: Never mention your role, simply generate the questions."""

def get_HKDSE_tutor_prompt():
    return f"""Role: You are an HKDSE examiner and curriculum specialist. 
    You have written actual HKDSE questions for the Hong Kong Examinations and Assessment Authority (HKEAA).
    You understand the exact style, difficulty level, and marking scheme requirements. You are able to access curated RAG retrieval from past HKDSE papers to ensure the questions you generate are fully aligned with the official standards. 
    Background: I am an HKDSE student.
    Constraints: -Questions must be original — do not copy past paper questions verbatim.
    Difficulty should start easy, then progress to medium — no HKDSE hard questions initially.
    NEVER show the correct or incorrect answers in the marking scheme when generating questions. The marking scheme should only show the steps required to get full marks, without revealing the final answer. 
    Never mention your role, simply generate the questions."""

def get_tutor_feedback_prompt(questions, user_answers):
    return f"""Here are the questions: {questions}\n\nUser's answers: {user_answers}\n\n
    Role: You are an experienced exam marker who has graded thousands of exam papers. You know exactly why students lose marks. You are patient and explain in a way that helps students learn, not just get the answer. 
    Background: I am a student who attempted a practice question. I need to understand the quality of my answers, WHAT the correct steps are, and if my answers are wrong, how can I avoid this mistake next time.
    Task: You will be given: The original question. The student's answer. 
    If the student is wrong, explain to the student why their answer is wrong, show the correct step-by-step working, and highlight the common mistake they made.
    Constraints:
- Start with a one-sentence empathetic statement (e.g., "Good try! Many students make this mistake.")
- Then state EXACTLY what the student did wrong
- Show the correct solution in numbered steps
- List 1-2 similar common mistakes students make on this question type
- End with a short practice tip
- Keep total explanation between 150-250 words
- Do not be sarcastic or negative — always encouraging

Example output:

Question: "Solve x² + 6x - 3 = 0 by completing the square."
Student answer: "x² + 6x - 3 = 0 → x² + 6x = 3 → (x+3)² = 3 → x = -3 ± √3"
Correct answer: "x = -3 ± 2√3"
Marking scheme excerpt: "Step 2: Add (b/2)² = 9 to both sides → x²+6x+9 = 3+9 → (x+3)² = 12 (2 marks)"

Output:
"Good try! Many students forget to add (b/2)² to BOTH sides. You correctly moved the constant term, but you forgot to add 9 to the right side. Let me show you:

Correct steps:
1. x² + 6x - 3 = 0 → x² + 6x = 3
2. Add (6/2)² = 9 to both sides: x² + 6x + 9 = 3 + 9
3. (x + 3)² = 12
4. x + 3 = ±√12 = ±2√3
5. x = -3 ± 2√3

Common mistake: Students often only add to the left side or forget to simplify √12 to 2√3.

Practice tip: Always write the step 'add (b/2)² to BOTH sides' explicitly to remind yourself."
Example output ends.
Now provide your explanation."""

def get_HKDSE_tutor_feedback_prompt(questions, user_answers):
    return f"""Here are the HK DSE {st.session_state.hkdse_category} questions: {questions}\n\nUser's answers: {user_answers}\n\n
    Role: You are an experienced HKDSE marker who has graded thousands of exam papers. You know exactly why students lose marks and what the official marking scheme requires. You are patient and explain in a way that helps students learn, not just get the answer.
    Background: I am an HKDSE student who attempted a practice question. I need to understand the quality of my answers, WHAT the correct steps are, and if my answers are wrong, how can I avoid this mistake next time.
    Task: You will be given: The original question. The student's answer. The relevant marking schemes (provided via RAG from past papers)
    If the student is wrong, explain to the student why their answer is wrong, show the correct step-by-step working, reference the official marking scheme, and highlight the common mistake they made.
    You are able to access curated RAG retrieval from past HKDSE papers to ensure your feedback is fully aligned with the official standards and standard marking schemes.
    Constraints:
- Start with a one-sentence empathetic statement (e.g., "Good try! Many students make this mistake.")
- Then state EXACTLY what the student did wrong
- Show the correct solution in numbered steps
- Quote the relevant part of the marking scheme (e.g., "According to the 2022 HKDSE marking scheme, step 2 is worth 2 marks...")
- List 1-2 similar common mistakes students make on this question type
- End with a short practice tip
- Keep total explanation between 150-250 words
- Do not be sarcastic or negative — always encouraging

Example output:

Question: "Solve x² + 6x - 3 = 0 by completing the square."
Student answer: "x² + 6x - 3 = 0 → x² + 6x = 3 → (x+3)² = 3 → x = -3 ± √3"
Correct answer: "x = -3 ± 2√3"
Marking scheme excerpt: "Step 2: Add (b/2)² = 9 to both sides → x²+6x+9 = 3+9 → (x+3)² = 12 (2 marks)"

Output:
"Good try! Many students forget to add (b/2)² to BOTH sides. You correctly moved the constant term, but you forgot to add 9 to the right side. Let me show you:

Correct steps:
1. x² + 6x - 3 = 0 → x² + 6x = 3
2. Add (6/2)² = 9 to both sides: x² + 6x + 9 = 3 + 9
3. (x + 3)² = 12
4. x + 3 = ±√12 = ±2√3
5. x = -3 ± 2√3

According to the marking scheme, step 2 (adding 9 to both sides) earns 2 marks — you would have lost those marks.

Common mistake: Students often only add to the left side or forget to simplify √12 to 2√3.

Practice tip: Always write the step 'add (b/2)² to BOTH sides' explicitly to remind yourself."
Example output ends.
Now provide your explanation.
"""

st.set_page_config(page_title="AI Study Tutor", page_icon="🎓")

st.title("🎓 AI Study Tutor")

# Sidebar for page selection
st.sidebar.info("Click on the buttons to navigate")

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
    uploaded_files = st.file_uploader("Upload your study notes (PDF)", type=["pdf"], accept_multiple_files=True, max_upload_size=5)

    # Custom prompt for question generation
    custom_prompt = st.text_area("Custom prompt (optional)", placeholder="Enter any additional instructions for question generation", key="custom_prompt_textarea")
    st.session_state.custom_prompt = custom_prompt

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
                        prompt = f"{get_tutor_prompt()} Based on this document, generate {st.session_state.quantity} multiple choice study questions. Each question should have 4 answer options labeled a, b, c, d. Do not provide any answers. {get_prompt_formatting_string()}"
                    elif st.session_state.question_type == "True or False":
                        prompt = f"{get_tutor_prompt()} Based on this document, generate {st.session_state.quantity} true or false study questions. Do not provide any answers. {get_prompt_formatting_string()}"
                    else:
                        prompt = f"{get_tutor_prompt()} Based on this document, generate {st.session_state.quantity} {st.session_state.question_type.lower()} study questions. Do not provide any answers. {get_prompt_formatting_string()}"

                    # Append custom prompt if provided
                    if st.session_state.custom_prompt:
                        prompt += f"\n\nAdditional instructions: {st.session_state.custom_prompt}"

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
        st.markdown(st.session_state.generated_questions)

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
                    feedback_prompt = get_tutor_feedback_prompt(st.session_state.generated_questions, answers)
                    
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
                    # Initialize conversation history with initial context
                    st.session_state.followup_conversation = (
                        f"Original questions: {st.session_state.generated_questions}\n"
                        f"User answers: {', '.join(answers)}\n"
                        f"AI feedback: {feedback_response}\n"
                        f"---\n"
                    )
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
                    # Append user input to conversation
                    st.session_state.followup_conversation += f"USER: {continue_text}  \n"
                    
                    # Get AI response with full conversation context
                    full_prompt = (
                        f"{st.session_state.followup_conversation}"
                        f"As a tutor, continue the conversation and provide helpful guidance based on all context above. Try to point out specific shortcomings of the user's answers and general mistakes (such as not thinking critically enough), and give specific advice on how to improve."
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
                    st.session_state.followup_conversation += f"AI: {ai_response}\n\n"
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
            ["Chinese", "English", "Math", "Physics", "Chemistry", "Biology", "Economics", "Business, Accounting, and Financial Studies", "Tourism and Hospitality Studies"],
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
        hkdse_uploaded_files = st.file_uploader("Upload study materials (optional, PDF)", type=["pdf"], accept_multiple_files=True, max_upload_size=5, key="hkdse_file_uploader")

        # Custom prompt for question generation
        hkdse_custom_prompt = st.text_area("Custom prompt (optional)", placeholder="Enter any additional instructions for question generation", key="hkdse_custom_prompt_textarea")
        st.session_state.hkdse_custom_prompt = hkdse_custom_prompt

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
                            prompt = f"{get_HKDSE_tutor_prompt()} Based on the uploaded materials, generate {st.session_state.hkdse_quantity} multiple choice questions for HK DSE {st.session_state.hkdse_category}. Each question should have 4 answer options labeled a, b, c, d. Do not provide any answers. {get_prompt_formatting_string()}"
                        else:
                            prompt = f"{get_HKDSE_tutor_prompt()} Generate {st.session_state.hkdse_quantity} multiple choice questions for HK DSE {st.session_state.hkdse_category}. Each question should have 4 answer options labeled a, b, c, d. Do not provide any answers. {get_prompt_formatting_string()}"
                    elif st.session_state.hkdse_question_type == "True or False":
                        if pdf_parts:
                            prompt = f"{get_HKDSE_tutor_prompt()} Based on the uploaded materials, generate {st.session_state.hkdse_quantity} true or false questions for HK DSE {st.session_state.hkdse_category}. Do not provide any answers. {get_prompt_formatting_string()}"
                        else:
                            prompt = f"{get_HKDSE_tutor_prompt()} Generate {st.session_state.hkdse_quantity} true or false questions for HK DSE {st.session_state.hkdse_category}. Do not provide any answers. {get_prompt_formatting_string()}"
                    else:
                        if pdf_parts:
                            prompt = f"{get_HKDSE_tutor_prompt()} Based on the uploaded materials, generate {st.session_state.hkdse_quantity} short answer questions for HK DSE {st.session_state.hkdse_category}. Do not provide any answers. {get_prompt_formatting_string()}"
                        else:
                            prompt = f"{get_HKDSE_tutor_prompt()} Generate {st.session_state.hkdse_quantity} short answer questions for HK DSE {st.session_state.hkdse_category}. Do not provide any answers. {get_prompt_formatting_string()}"

                    # Append custom prompt if provided
                    if st.session_state.hkdse_custom_prompt:
                        prompt += f"\n\nAdditional instructions: {st.session_state.hkdse_custom_prompt}"

                    # COMMENT THIS BACK IN WHEN YOU WANT TO TEST WITH THE MODEL
                    response = rag_model.generate_content(pdf_parts + [prompt] if pdf_parts else [prompt]).text
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
            st.markdown(st.session_state.hkdse_generated_questions)

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
                        feedback_prompt = get_HKDSE_tutor_feedback_prompt(st.session_state.hkdse_generated_questions, hkdse_answers)
                        
                        # COMMENT THIS BACK IN WHEN YOU WANT TO TEST WITH THE MODEL
                        feedback_response = rag_model.generate_content(feedback_prompt).text

                        # if feedback_response.candidates[0].grounding_metadata:
                        #     metadata = feedback_response.candidates[0].grounding_metadata
                        #     print("\nSources used for feedback:")
                            
                        #     if metadata.grounding_chunks:
                        #         for i, chunk in enumerate(metadata.grounding_chunks):
                        #             source_name = chunk.retrieved_context.uri.split('/')[-1]
                        #             print(f"[{i+1}] {source_name}")
                        # feedback_response = feedback_response.text

                        # COMMENT THIS BACK IN WHEN YOU DONT WANT TO TEST WITH THE MODEL
                        # time.sleep(3)
                        # feedback_response = f"Feedback for answers: {', '.join(hkdse_answers)} - This is mock feedback."
                        
                        st.session_state.hkdse_user_answers = hkdse_answers
                        st.session_state.hkdse_feedback_response = feedback_response
                        st.session_state.hkdse_answers_submitted = True
                        # Initialize conversation history with initial context
                        st.session_state.hkdse_followup_conversation = (
                            f"Original questions: {st.session_state.hkdse_generated_questions}\n"
                            f"User answers: {', '.join(hkdse_answers)}\n"
                            f"AI feedback: {feedback_response}\n"
                            f"---\n"
                        )
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
                        # Append user input to conversation
                        st.session_state.hkdse_followup_conversation += f"USER: {continue_text}  \n"
                        
                        # Get AI response with full conversation context
                        full_prompt = (
                            f"{st.session_state.hkdse_followup_conversation}"
                            f"As a tutor, continue the conversation and provide helpful guidance based on all context above. Try to point out specific shortcomings of the user's answers and general mistakes (such as not thinking critically enough), based on the official HKDSE marking scheme, and give specific advice on how to improve according to HKDSE standards."
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
                        st.session_state.hkdse_followup_conversation += f"AI: {ai_response}\n\n "
                        st.session_state.hkdse_followup_response = ai_response

                    latest_hkdse_ai = st.session_state.hkdse_followup_response if st.session_state.hkdse_followup_response else st.session_state.hkdse_feedback_response
                    if latest_hkdse_ai:
                        st.markdown("### Latest AI Response")
                        st.markdown(latest_hkdse_ai)

                    if st.session_state.hkdse_followup_conversation:
                        st.markdown("### Conversation History")
                        st.markdown(st.session_state.hkdse_followup_conversation)
