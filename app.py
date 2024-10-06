import os
import dotenv
import requests
from flask import Flask, render_template, request, jsonify
from PIL import Image
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import mimetypes
import boto3
from langchain_core.prompts import PromptTemplate
from langchain_aws import BedrockLLM

# Load environment variables
dotenv.load_dotenv()

# Azure Search configuration
AZURE_SEARCH_ENDPOINT = "https://visionrag.search.windows.net"
AZURE_SEARCH_IMAGES_INDEX = "images-index-poc-2"
# Azure Computer Vision configuration
AZURE_COMPUTER_VISION_URL = "https://visionrag2.cognitiveservices.azure.com/computervision/retrieval"

AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_COMPUTER_VISION_KEY = os.getenv("AZURE_COMPUTER_VISION_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Initialize Flask app
app = Flask(__name__)

# Initialize Azure Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_IMAGES_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

def load_llm():
    try:
        # Create a Bedrock client
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name='us-east-1',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        model_id = "meta.llama3-70b-instruct-v1:0"

        model_kwargs = { 
            "temperature": 0.01,
            "top_p": 0.9,
            "max_gen_len": 250
        }
        llm = BedrockLLM(
            client=bedrock_runtime,
            model_id=model_id,
            model_kwargs=model_kwargs,
            streaming=True
        )
        return llm
    except Exception as e:
        print(f"Error loading LLM: {e}")
        return None

def llm_response(query):
    try:
        llm = load_llm()
        if llm is None:
            return "No relevant images found. Enter a valid query."

        prompt =  """ 
            You are a very intelligent AI assistant who is an expert in checking the context of the user query. The user query should match any alcohol/wine bottles related questions, keywords, pictures and logos. Make sure you understand the typos.
            If they match return "Yes".
            If they do not match the context return "No relevant images found. Enter a valid query". 

            Make sure your response is only one of the above two words.

            For example:

            question: "wine and tree"
            answer: "Yes"

            question: "whats the capital of New Jersey?"
            answer: "No relevant images found. Enter a valid query"

            Input: {query}
            Output:
            """

        query_with_prompt = PromptTemplate(
            template=prompt,
            input_variables=["query"]
        )
        
        llmchain = query_with_prompt | llm 
        response = llmchain.invoke({"query": query}) 
        return response
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return "No relevant images found. Enter a valid query."

# Function to get image embedding from Azure Computer Vision
def get_model_params():
    return {"api-version": "2023-02-01-preview", "modelVersion": "latest"}

def get_auth_headers():
    return {"Ocp-Apim-Subscription-Key": AZURE_COMPUTER_VISION_KEY}

def get_image_embedding(image_file):
    try:
        mimetype = mimetypes.guess_type(image_file)[0]
        url = f"{AZURE_COMPUTER_VISION_URL}:vectorizeImage"
        headers = get_auth_headers()
        headers["Content-Type"] = mimetype
        with open(image_file, "rb") as f:
            response = requests.post(url, headers=headers, params=get_model_params(), data=f)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()}")
            return None
        return response.json()["vector"]
    except Exception as e:
        print(f"Error getting image embedding for {image_file}: {e}")
        return None

def get_text_embedding(text):
    try:
        url = f"{AZURE_COMPUTER_VISION_URL}:vectorizeText"
        response = requests.post(url, headers=get_auth_headers(), params=get_model_params(), json={"text": text})
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()}")
            return None
        return response.json()["vector"]
    except Exception as e:
        print(f"Error getting text embedding for '{text}': {e}")
        return None

# Function to add image vectors to search index
def index_images():
    try:
        for image_file in os.listdir("images"):
            image_embedding = get_image_embedding(f"images/{image_file}")
            if image_embedding:
                search_client.upload_documents(documents=[{
                    "id": image_file.split(".")[0],
                    "filename": image_file,
                    "embedding": image_embedding
                }])
    except Exception as e:
        print(f"Error indexing images: {e}")

# Function to search images based on a query
def search_images(query):
    try:
        query_vector = get_text_embedding(query)
        if not query_vector:
            return None

        # Perform vector search
        results = search_client.search(
            search_text=None,
            vector_queries=[VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=3,
                fields="embedding"
            )]
        )
        
        all_results = [doc["filename"] for doc in results]
        images = []
        
        for result in all_results:
            print(result)
            img_path = f"static/images/{result}"
            try:
                images.append(img_path)
            except Exception as e:
                print(f"Error adding image path for {result}: {e}")
        
        return images if images else None
    except Exception as e:
        print(f"Error searching images for query '{query}': {e}")
        return None

# Route for home page
@app.route('/')
def home():
    return render_template('chat.html')

# Route for handling user queries
@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_query = request.form['message']
        allow_image_retrieval = llm_response(user_query)
        
        if 'No relevant' not in allow_image_retrieval:
            image_response = search_images(user_query)
            if image_response:
                return jsonify({'images': image_response, 'message': 'Here are my best three suggestions from left to right'})
        
        return jsonify({'message': 'No relevant images found. Enter a valid query.'})
    except Exception as e:
        print(f"Error handling user query: {e}")
        return jsonify({'message': 'An error occurred while processing your request.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  

