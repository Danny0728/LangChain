# langchain imports
# from langchain.text_splitter import RecursiveCharacterTextSplitter deprecated
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
# from langchain.vectorstores import FAISS deprecated

# langchain community imports
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
# langchain OpenAi imports
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# text_splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
# others 
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

def create_vector_db_from_yt_url(video_url:str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcipt = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)

    docs = text_splitter.split_documents(transcipt)

    db = FAISS.from_documents(docs,embeddings)
    return db

def get_response_from_query(db: FAISS,query,k=4): 
    # we will use gpt-4o-mini cause it is cost effective
    docs = db.similarity_search(query,k=k)

    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = PromptTemplate(
        input_variables=['question','docs'],
        template="""
        You are a helpful Youtube assitant that can answer questions about  videos based on the video's transcript.

        Answer the following question: {question}
        By sesarching the following video transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I dont't know"

        Your answers should be detailed."""
    )

    chain = prompt | llm
    
    response = chain.invoke({'question':query, 'docs':docs_page_content})
    print(response.content)
    return response.content ,docs
