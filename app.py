import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
  text = ""
  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      page.extract_text()
      text += page.extract_text()
  return text

def get_text_chunks(text):
  text_splitter = CharacterTextSplitter(
    separator = '\n',
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
  )
  
  chunks = text_splitter.split_text(text)
  return chunks

def get_conversation_chain(vectorstore):
  llm = ChatOpenAI()
  memory = ConversationBufferMemory(memory_key= 'chat_history', return_messages=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(
    llm = llm, 
    retriever=vectorstore.as_retriever(),
    memory=memory
  )
  return conversation_chain
  

def get_vectorstore(text_chunks):
  embeddings = OpenAIEmbeddings()
  # embeddings = HuggingFaceInstructEmbedding(model_name='instructor-xl')
  vectorstore = FAISS.from_texts(texts = text_chunks, embedding=embeddings)
  return vectorstore

def main():
  load_dotenv()
  st.set_page_config(page_title='Chat with multiple PDFs', page_icon= ':books:')
  
  st.write(css, unsafe_allow_html=True)
  st.header('Chat with multiple PDFs :books:')
  st.text_input('Ask away:')
  
  # reinitialise the session state
  if 'conversation' not in st.session_state:
    st.session_state.conversation = None
  # st.session_state.conversation
  
  st.write(user_template.replace("{{MSG}}", 'Hello Bot'), unsafe_allow_html=True)
  st.write(bot_template.replace("{{MSG}}", 'Hello User'), unsafe_allow_html=True)
  
  with st.sidebar:
    st.subheader('Your documents')
    pdf_docs = st.file_uploader('Upload your PDFs here', accept_multiple_files=True)
    if st.button('Process'):
      with st.spinner("Processing"):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        st.write(text_chunks)
        
        vectorstore = get_vectorstore(text_chunks)
        # allow us to generate new messages
        # takes history and gives next element of convo
        st.session_state.conversation = get_conversation_chain(vectorstore)
        #  variable linked to session - persistent


if __name__ == '__main__':
  main()