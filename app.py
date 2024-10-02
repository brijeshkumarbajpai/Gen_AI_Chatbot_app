## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import os


LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="My__Gen_AI_Chatbot_APP"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## set up Streamlit 
st.title("Hi, I am Helplio!\nHow can I help you?")
# st.write("Please upload Pdf's")

llm=ChatGroq(groq_api_key=Groq_API_Key,model_name="Gemma2-9b-It")

## chat interface
name=st.text_input("Please Enter Your Name:")
if name:
    st.write(f"Hi, {name}, Nice to meet you 🙂")
    session_id=st.text_input("Your Session ID",value=f"{name}@123")

    ## statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store={}

    if session_id:
        uploaded_files=st.file_uploader(f" Please choose A Pdf file.",type="pdf",accept_multiple_files=True)
        if uploaded_files:
            documents=[]
            for uploaded_file in uploaded_files:
                temppdf=f"./temp.pdf"
                with open(temppdf,"wb") as file:
                   file.write(uploaded_file.getvalue())
                   file_name=uploaded_file.name
                loader=PyPDFLoader(temppdf)
                docs=loader.load()
                documents.extend(docs)
            if not documents:
               st.error("No documents were loaded.")
               st.stop()
            #Split and create embeddings for the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            if not splits:
               st.error("No document splits were generated.")
               st.stop()
            # Check if embeddings are being generated correctly
            try:
              test_embedding = embeddings.embed_query("Test sentence")
            #   st.write("Test embedding generated successfully.")
            except Exception as e:
               st.error(f"Embedding generation failed: {e}")
               st.stop()
    
            # Create the FAISS vectorstore
            try:
              vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            #   st.write("FAISS vector store created successfully.")
            except Exception as e:
              st.error(f"FAISS vector store creation failed: {e}")
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
            )
            history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
            
            # Answer question
            system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
            )
            question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
            # Reg Chain
            rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

            def get_session_history(session:str)->BaseChatMessageHistory:
               if session_id not in st.session_state.store:
                 st.session_state.store[session_id]=ChatMessageHistory()
               return st.session_state.store[session_id]
            conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
            )

            user_input = st.text_input("Your question:")

            if user_input:
                session_history=get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
                )
                # st.write(st.session_state.store)
                st.write("Assistant:", response['answer'])
                st.write("Chat History:", session_history.messages)