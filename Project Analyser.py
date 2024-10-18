import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import json
import boto3  # For AWS S3 integration (optional for cloud storage)

# Load API key safely
try:
    from key import GOOGLE_API_KEY  # Ensure the key file exists

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except ImportError:
    st.error("API Key not found. Please provide a valid Google API Key.")
    st.stop()

# Initialize Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Enhanced prompt for grading and feedback
def get_conversational_chain():
    prompt_template = """
    You are an expert assistant specialized in reviewing, grading, and debugging projects in various formats, including but not limited to Python, HTML, JavaScript, C++, C#, and PDFs. 
    You are tasked with:
    - Analyzing the provided project files for correctness and adherence to best practices.
    - Identifying potential bugs, errors, and areas of improvement within the code or content.
    - Providing constructive feedback on the overall structure, logic, and style of the project.
    - Suggesting specific solutions or code improvements for any issues found.
    - If the project is incomplete or incorrect, generate the entire corrected or improved version of the project, explaining the changes step by step.

    Here is your task:
    1. *Review the provided project*: This includes analyzing code, structure, and formatting.
    2. *Grade the project out of 10* based on correctness, code quality, structure, efficiency, and adherence to best practices.
    3. *Provide constructive feedback*: Clearly explain any mistakes, and offer specific improvements or code suggestions.
    4. *Generate a corrected project* if the current version is incorrect or incomplete. Provide the entire updated project with explanations for all major changes.

    Context:
    {context}

    Question:
    {question}

    Detailed Answer:
    """

    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error loading conversational chain: {e}")
        return None


# Function to break code blocks into lists of lines for better formatting in JSON


# Function to output JSON with grade
def output_grade(grade):
    grade_data = {"grade": grade}
    with open('grade.json', 'w') as json_file:
        json.dump(grade_data, json_file)
    st.success("Grade saved to grade.json")


# Function to extract text from various files (PDF, text, code)
def get_file_text(file_docs):
    text = ""
    for file in file_docs:
        try:
            file_extension = file.name.split('.')[-1].lower()

            # Handle PDF files
            if file_extension == "pdf":
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()

            # Handle plain text files or programming files
            elif file_extension in ["py", "html", "static", "js", "java", "cpp", "c", "cs", "txt"]:
                text += file.read().decode('utf-8')  # Assuming files are UTF-8 encoded

            else:
                st.warning(f"Unsupported file format: {file.name}")

        except Exception as e:
            st.error(f"Error processing file {file.name}: {e}")

    return text


# Split text into chunks for easier processing
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []


# Save the vector store to AWS S3 (optional)
def save_to_s3(file_name, bucket_name):
    try:
        s3 = boto3.client('s3')
        s3.upload_file(file_name, bucket_name, file_name)
        st.success(f"Vector store saved to S3 bucket: {bucket_name}")
    except Exception as e:
        st.error(f"Error uploading to S3: {e}")


# Create FAISS vector store and optionally save it to S3
def get_vector_store(text_chunks):
    try:
        embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')  # Ensure correct model
        vector_store = FAISS.from_texts(text_chunks, embedding=embedding)
        vector_store.save_local("faiss_index")

        # Optionally save to cloud (AWS S3)
        bucket_name = os.getenv("AWS_BUCKET_NAME")
        if bucket_name:
            save_to_s3("faiss_index", bucket_name)

        return vector_store
    except Exception as e:
        st.error(f"Error embedding content or creating vector store: {e}")
        return None


# Process user input question and save JSON outputs
# Process user input question and save JSON outputs# Process user input question and generate grade JSON file at the top
import re

# Process user input question and generate grade JSON file based on model's output
def user_input(user_question):
    try:
        embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        vector_store = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)

        if not docs:
            st.error("No relevant documents found.")
            return

        chain = get_conversational_chain()
        if chain is None:
            return

        # Run the chain to get the response
        response = chain({
            "input_documents": docs,
            "question": user_question
        }, return_only_outputs=True)

        # Extract the answer (including grade and feedback)
        answer = response.get("output_text", "No response generated.")
        feedback = answer.strip()  # Clean up unnecessary spaces

        # Use regex to find the grade (e.g., "Grade: 7/10")
        match = re.search(r'Grade:\s*(\d+)/10', feedback)
        if match:
            grade = int(match.group(1))
        else:
            grade = 5  # Default grade if no grade is found in the response

        # Output the grade and feedback
        st.write(f"**Grade:** {grade}/10")
        st.write(f"**Feedback:** {feedback}")

        # Save the grade to a JSON file
        output_grade(grade)

    except Exception as e:
        st.error(f"Error processing user input: {e}")




# Main UI for Chatbot with Session State handling for file uploads
def main():
    st.set_page_config(page_title="QA Chatbot with Files (PDF/Code) using Gemini", page_icon="💁")
    st.header("QA Chatbot with Files (PDF or Code) using Gemini AI 💁")

    # File uploader for PDFs and programming files with session state handling
    with st.sidebar:
        st.title("Upload Files")
        if "file_docs" not in st.session_state:
            st.session_state.file_docs = None

        file_docs = st.file_uploader("Upload your PDF or Programming Files", accept_multiple_files=True)
        if st.button("Submit & Process") and file_docs:
            st.session_state.file_docs = file_docs  # Store files in session state
            with st.spinner("Processing..."):
                raw_text = get_file_text(st.session_state.file_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        vector_store = get_vector_store(text_chunks)
                        if vector_store:
                            st.success("Files processed and vector store saved locally. You can now ask questions.")
        elif st.session_state.file_docs:
            st.success("Files already uploaded and processed. You can ask questions.")

    # Chatbot interface for question-answering
    if st.session_state.file_docs:  # Ensure files are persisted across interactions
        user_question = st.text_input("Ask a question based on the uploaded files:")
        if st.button("Ask") and user_question:
            user_input(user_question)
    else:
        st.warning("Please upload and process files before asking questions.")


if __name__ == "__main__":
    main()
