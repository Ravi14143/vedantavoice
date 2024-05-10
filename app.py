import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from googletrans import Translator
from langdetect import detect
from typing import Dict, Any
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
from urllib.parse import quote

# Define a function to embed PDF content
def embed_pdf(pdf_url):
    return f'<iframe src="https://docs.google.com/viewer?url={quote(pdf_url)}&embedded=true" style="width:100%; height:600px;" frameborder="0"></iframe>'


DB_FAISS_PATH = 'vectorstore/db_faiss/'

custom_prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 2}), return_source_documents=True, chain_type_kwargs={'prompt': prompt})
    return qa_chain

def load_llm():
    llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama", max_new_tokens=512, temperature=0.5)
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # llm=load_llm()
    # chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
    #                                             retriever=FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k":2}),
    #                                             memory=memory)
    # return chain

def translate_text(text, dest_lang):
    translator = Translator()
    translation = translator.translate(text, dest=dest_lang)
    return translation.text

def final_result(query, dest_lang="en", user_lang="auto"):
    if user_lang == "auto":
        user_lang = detect(query)

    if user_lang != dest_lang:
        translated_query = translate_text(query, dest_lang)
    else:
        translated_query = query

    qa_result = qa_bot()
    response = qa_result(translated_query)

    if user_lang != dest_lang:
        translated_response = translate_text(response['result'], user_lang)
        response['result'] = translated_response

    return response


# from whisperspeech.pipeline import Pipeline
# pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')
# s = pyttsx3.init()
def main():

    st.title("Vedanta Voice")
    def conversation_chat(query):
        result = final_result(query)
        st.session_state['history'].append((query, result['result']))
       # query = "What we have to learn from the Bhagavad Gita"
       # result = final_result(query)
        print(result)
        #pipe.generate_to_notebook(result['result'], lang='en', speaker='/content/drive/MyDrive/prabhupadha_voice.mp3')
        return result

    def initialize_session_state():
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me"]

        if 'source' not in st.session_state:
            st.session_state['source'] = ["here are source"]

        if 'links' not in st.session_state:
            st.session_state['links'] = ["here are links"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]

    def display_chat_history():
        reply_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Question:", placeholder="Talk with Krishna", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversation_chat(user_input)
                print(output)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output['result'])
                st.session_state['source'].append(output['source_documents'])
                print(output['source_documents'])
                #print(output['source_documents'].metadata)
                for i in output['source_documents']:
                   st.session_state['links'].append(i.metadata['source'])


        print(st.session_state['generated'])
        if st.session_state['generated']:
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style='gamer.png')


                    source_links = st.session_state['links'][i]
                    source_link='/content/drive/MyDrive/data/'+source_links.split('\\')[-1]
                    print(source_link)

                    # Convert source_link to an HTML link
                    source_link_html = f'<a href="{quote(source_link)}" target="_blank">Source Documents</a>'

                    # Embed PDF content
                    pdf_embed_html = embed_pdf(source_link_html)


                    combined_message = f"{st.session_state['generated'][i]}\n"


                    message(combined_message, key=str(i), avatar_style="krishna.png")

                    st.markdown(source_link_html, unsafe_allow_html=True)
                    if st.button('Audio', key=f'audio_button_{i}'):
                      pass
                        # s.say(st.session_state['generated'][i])
                        # s.runAndWait()


                    # try:
                    #     audio_path = pipe.generate_to_notebook(st.session_state['generated'][i], lang='en', speaker='/content/drive/MyDrive/prabhupadha_voice.mp3')
                    #     st.audio(audio_path, format='audio/mp3', start_time=0)
                    # except Exception as e:
                    #     st.error(f"Error generating audio: {e}")

                    #message(pipe.generate_to_notebook(st.session_state["generated"][i], lang='en', speaker='/content/drive/MyDrive/prabhupadha_voice.mp3'))

    # Initialize session state
    initialize_session_state()
    # Display chat history
    display_chat_history()

if __name__ == '__main__':
    main()
