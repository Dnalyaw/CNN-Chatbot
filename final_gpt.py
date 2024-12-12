import os
import tiktoken
import langchain
import numpy as np
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#from .autonotebook import tqdm as notebook_tqdm
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm, trange
import spacy
import pandas as pd

import warnings
warnings.filterwarnings("ignore") 


#Wayland's Personal API Key 
os.environ['OPENAI_API_KEY'] = openai_key
gpt_llm = OpenAI(temperature=1.5, max_tokens=500, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.5)

#in order to figure figure out how many tokens OpenAI uses, we are going to calculate it
tokenizer = tiktoken.get_encoding('cl100k_base') #this is the type of token splitting that openai uses, which changes on the llm (gpt2, gpt3, etc.)

#len function  (how many tokens will i have to pay to do this)
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text, 
        disallowed_special=()
    )
    return len(tokens)
tiktoken.encoding_for_model('gpt-3.5-turbo')#check the token calculator using this

#the original Dataset we are using
articles = pd.read_csv('20241018-235356\CNN_Articels_clean\CNN_Articels_clean.csv')

#load data
nlp = spacy.load("en_core_web_sm")
# Load a pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

#removing stopwords from query
def stopword_removal(text):
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text

print("\n\n\n\n\n\n\n\n")
print("What type of topic do you want to ask CNN-GPT? (Ex: Tell me about the Golden State Warriors.)")
#query = "Tell me about the Golden State Warriors."
query = input("Response: ")
print('Currently Finding CNN Articles Pertaining to response...')
disasters = articles[articles["Keywords"].apply(lambda x: any(i in x for i in stopword_removal(query).split(" ")))]

#these give the top 5 best CNN sites that pertain to the topic at hand
vectors = model.encode(query)
disaster_matrix = np.vstack(disasters["Description"].apply(lambda x: model.encode(x)).to_numpy())
indices = np.argsort(np.dot(disaster_matrix, vectors)/(np.linalg.norm(disaster_matrix, axis = 1)*np.linalg.norm(vectors)))[-1:-5:-1]
best_fit = disasters.iloc[indices]
best_fits = list(best_fit.head()['Url'])

print("Best Fit Articles: " + str(best_fits))


#loading the data into GPT
print("Now loading data into Chatbot environment...")

loaders = UnstructuredURLLoader(
    urls = best_fits
)  
data = loaders.load()


#splitting the data 
print("Chunking data...")
token_counts = [tiktoken_len(doc.page_content) for doc in data]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, #max size of the length
    chunk_overlap=20, #because splitting might take awa some context, some of the chunks will overlap to maintain this context
    length_function=tiktoken_len, #finds out the amount of tokens used in these files
    separators=['\n\n','\n', '.', ' '] #splits based on the separators, if too long will try and split words based on this order
)

chunks = text_splitter.split_documents(data)

print('Embedding data...')
#converts chunks into vectorindex, kind of like a vector database
embeddings = OpenAIEmbeddings()
vectorindex_openai = FAISS.from_documents(chunks, embeddings)

#the GPT is created!
chain = RetrievalQAWithSourcesChain.from_llm(llm=gpt_llm, retriever=vectorindex_openai.as_retriever())

#use debug to see what is going on behind the hood
#langchain.debug=True
print("\nCNN-GPT created! Ask any information you want! (Ex. Why are the Golden State Warriors such a good team?)")


gpt_prompt = input('Prompt: ')
while gpt_prompt != 'q':
    response = chain({'question': gpt_prompt + " Be as descriptive and creative as possible"})

    print('\nAnswer:')
    print(response['answer'] + "\n From Source: " + response['sources'])
    gpt_prompt = input('New Prompt (type q to quit): ')
