#Importing neccesary libraries
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import StuffDocumentsChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tabula
import os

#Setting up our env
# from dotenv import load_dotenv
# load_dotenv()

#Getting the Hugging Face Token and GroqApiKey
# os.environ['HF_TOKEN']=st.secrets['HF_TOKEN']
groq_api_key=st.secrets['GROQ_API_KEY']
# embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#Setting up streamlit
st.title("Identifying Red flags in a SAAS document")
st.write("Upload pdf to see the red flags")
temperature = 0.7

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-70b-versatile",temperature=temperature)

#Uploading the pdf
uploaded_documents=st.file_uploader("Choose A Pdf file to upload",type="pdf",accept_multiple_files=True)
if uploaded_documents:
    documents=[]
    for uploaded_document in uploaded_documents:
        temp_pdf=f"./temp.pdf" 
        with open(temp_pdf,'wb') as file:
            file.write(uploaded_document.getvalue())
            file_name=uploaded_document.name
        loader=PyPDFLoader(temp_pdf)
        docs=loader.load()
        documents.extend(docs)


    
    text_splitter3=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=600)
    chunks3=text_splitter3.split_documents(documents) 
    # vector_store=FAISS.from_documents(documents=chunks3,embedding=embeddings)
    # faiss_semantic_retriever=vector_store.as_retriever()
    # bm25_retriever = BM25Retriever.from_documents(documents=chunks3)
    # retriever = EnsembleRetriever(retrievers=[bm25_retriever,faiss_semantic_retriever],
    #                                    weights=[0.5,0.5])

    system_prompt = (
    "You are an assistant specialized in identifying red flags for the customer."
    "Your task is to help people make an informed decision by analyzing the terms and conditions (T&C) of SaaS contracts."
    "You will be given a piece of information from the T&C document, and you are required to identify and flag potential risks or unfavorable clauses for the customer."
    "Only highlight those clauses that a customer must be aware of before signing the deal, especially those that may have legal, financial, or operational implications."
    "Below are key areas you should pay attention to when analyzing the document:"
    
    "\n"
    "License Scope - Flexible and clear usage rights to ensure the customer can use the software for internal business operations without limitations, including subsidiaries and affiliates."
    "Data Ownership - Customer retains full ownership of their data, with the right to download, migrate, or delete it at any time without penalties."
    "Payment Term - Option to pay monthly or annually in arrears, with clear provisions for dispute resolution and extended payment terms (e.g., net 60). Avoid hidden penalties or interest."
    "Service Level Agreement (SLA) - Service credits for excessive downtime, strong guarantees for service uptime (e.g., 99.9%), and the right to terminate after repeated SLA breaches."
    "Data Protection - Ensure that customer data is encrypted both at rest and in transit, with strong contractual obligations on the provider to comply with industry standards and regulations."
    "Support - Ensure inclusion of 24/7 customer support channels and reasonable response times in case of technical issues, ensuring quick resolutions."
    "Termination Rights - Customer should have the right to terminate the agreement without penalties in cases of non-performance or breaches of key obligations by the provider."
    "Indemnification - Indemnity clauses should favor the customer, protecting them from IP infringement claims, with the SaaS provider bearing all legal costs."
    "Limitation of Liability - The limitation of liability should be capped at a reasonable amount (e.g., the total fees paid under the contract) and not exclude damages caused by negligence or data breaches."
    "Data Backup and Recovery - Customer's right to periodic data backups and assurances from the provider on robust disaster recovery plans."
    "Confidentiality - The agreement must include stringent confidentiality clauses to protect the customer’s sensitive information from unauthorized use or disclosure."
    "Audit Rights - The customer should have audit rights to ensure that the SaaS provider is complying with security, data protection, and other obligations."
    "Service Portability - Customers should be granted the ability to easily migrate data to a new provider or internal system if they choose to terminate the SaaS contract."
    "Automatic Renewal - Contracts should require explicit customer consent for renewals, with clear termination options before auto-renewal deadlines."
    "Warranties - The provider should warrant that the software will perform as advertised, meet specific performance standards, and provide remedies for failure to meet these warranties."
    "Dispute Resolution - Ensure clear, customer-friendly processes for dispute resolution, including mediation and arbitration, with jurisdiction chosen in favor of the customer."
    "Security Breaches - Immediate customer notification of any security breaches, along with comprehensive plans for remediation and compensation for damages."
    "Customizations - Customer’s right to request reasonable software customizations without incurring excessive fees or restrictions on implementation."
    
    "\n"
    "You should also watch out for any clauses that restrict customer rights, impose excessive fees, limit liability in unfair ways, or give undue advantage to the vendor."
    "Whenever a clause is found that may not be in the customer's best interest, provide a clear explanation of why it could pose a risk, and suggest how it could be improved if relevant."
    "Output only the clauses or issues that are important for the customer to be aware of before making a decision."
)
    qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{context}"),
                ]
            )
    # qa_chain = load_qa_chain(llm, chain_type="stuff")
    question_answer_chain=create_stuff_documents_chain(llm=llm,prompt=qa_prompt,document_variable_name="context")
    Whole_document=[]
    for i in range(len(chunks3)):
        doc=[Document(page_content=str(chunks3[i].page_content))]
        response = question_answer_chain.invoke({"context": doc})
        Whole_document.append(response)
    cleandocs=[]
    for i in range(len(Whole_document)):
        cleandocs.append(Document(page_content=str(Whole_document[i])))
    # schain=load_summarize_chain(
    # llm=llm,
    # chain_type="refine",
    # verbose=True    
    # )
    # output_summary=schain.run(cleandocs)
    # text=""

    val=8000//len(Whole_document)

    chunks_prompt="""
    Please summarize the below Terms and Condition Document in less than X words:
    Document:'{text}'
    Summary:

    """

    list1=[elm for elm in chunks_prompt]
    for i in range(len(list1)):
        if list1[i]=='X':
            list1[i]=val
    chunks_prompt1="".join([str(elm) for elm in list1])



    map_prompt_template=PromptTemplate(input_variables=['text'],template=chunks_prompt1)
    final_prompt='''
    Provide the red flags of the entire Terms and conditions that should be highlighted to the cutomer for taking an informed and good decision keeping the most important 20 solid points.
    Document:{text}

    '''
    final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)
    summary_chain=load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt_template,
    combine_prompt=final_prompt_template,
    verbose=True
    )
    output_summary=summary_chain.run(cleandocs)

    st.write(output_summary)
    # st.write(chunks_prompt)
