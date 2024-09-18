from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Custom prompt template for QA retrieval
custom_prompt_template = """As a medical expert, I'll provide advice based on the information you provide.
                                     Please describe the medical issue or question you have.
                                     If you don't know the answer, just say that you don't know, don't try to make
                                     up an answer.
Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    Set a custom prompt for the QA retrieval.
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt


def retrieval_qa_chain(llm, prompt, db):
    """
    Create a Retrieval QA chain for question answering.

    Args:
        llm (CTransformers): The language model, prompt (PromptTemplate): The prompt template for QA retrieval
        db (FAISS): The FAISS vector store.

    Returns:
        RetrievalQA: The Retrieval QA chain
    """
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt})
    return qa_chain


def load_llm():
    """
    Load the language model

    Returns:
        CTransformers: The loaded language model.
    """
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm


def qa_bot():
    """
    Initialize the QA bot for question answering.

    Returns:
        RetrievalQA: The QA bot.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'gpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


import re

# Confidence check function
def check_confidence(answer, question, context):
    """
    This function assesses the confidence level of the generated answer based on several heuristics:
    - Length of the answer
    - Presence of uncertainty phrases
    - Direct match with the context and question
    - Repetitive or vague content

    If the answer is considered low confidence, it returns "Data Not Available".
    """
    uncertainty_phrases = [
        "I'm not sure", "I think", "maybe", "possibly", "could be", "likely", 
        "probably", "it seems", "not certain"
    ]
    
    for phrase in uncertainty_phrases:
        if phrase.lower() in answer.lower():
            return "Data Not Available"
    
    if len(answer.split()) < 5:
        return "Data Not Available"
    

    if len(set(answer.split())) / len(answer.split()) < 0.5:
        return "Data Not Available"
    

    if question.lower() in context.lower() and question.lower() not in answer.lower():
        return "Data Not Available"
    

    if re.search(r"\b(it|that|this)\b", answer.lower()):
        return "Data Not Available"
    
    return answer

# Final result function with confidence check
def final_result(query):
    qa_result = qa_bot()  
    response = qa_result({'query': query})  
    
    answer = response["result"]
    sources = response["source_documents"]
    
    context = " ".join([doc.page_content for doc in sources])  
    confident_answer = check_confidence(answer, query, context)
    
    if confident_answer == "Data Not Available":
        return confident_answer
    else:
        if sources:
            confident_answer += f"\nSources:" + str(sources)
        else:
            confident_answer += "\nNo sources found"
        return confident_answer


# chainlit code
@cl.on_chat_start
async def start():
    """
    Start the chatbot.
    """
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    """
    Main function to handle user messages and provide responses.

    Args:
        message (cl.Message): The user's message.
    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()






