import os
import mimetypes
import requests
import math
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Azure Search and Computer Vision configurations
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_IMAGES_INDEX = os.getenv("AZURE_SEARCH_IMAGES_INDEX")
AZURE_COMPUTER_VISION_URL = os.getenv("AZURE_COMPUTER_VISION_URL")
AZURE_COMPUTER_VISION_KEY = os.getenv("AZURE_COMPUTER_VISION_KEY")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")


# Initialize Azure Search client
search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_IMAGES_INDEX, AzureKeyCredential(AZURE_SEARCH_KEY))

THRESHOLD = 0.243  # Define the similarity threshold
DIFFERENCE_THRESHOLD = 0.03

def get_model_params():
    return {"api-version": "2024-02-01", "model-version": "2022-04-11"}

def get_auth_headers():
    return {"Ocp-Apim-Subscription-Key": AZURE_COMPUTER_VISION_KEY}

def get_image_embedding(image_file):
    mimetype = mimetypes.guess_type(image_file)[0]
    url = f"{AZURE_COMPUTER_VISION_URL}:vectorizeImage"
    headers = get_auth_headers()
    headers["Content-Type"] = mimetype
    response = requests.post(url, headers=headers, params=get_model_params(), data=open(image_file, "rb"))
    
    if response.status_code != 200:
        print(image_file, response.status_code, response.json())
        return None
    
    return response.json()["vector"]

def get_text_embedding(text):
    url = f"{AZURE_COMPUTER_VISION_URL}:vectorizeText"
    headers = get_auth_headers()
    response = requests.post(url, headers=headers, params=get_model_params(), json={"text": text})
    
    if response.status_code != 200:
        print(f"Error getting text embedding for '{text}': {response.status_code}")
        return None

    return response.json()["vector"]

def get_cosine_similarity(vector1, vector2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(v ** 2 for v in vector1))
    magnitude2 = math.sqrt(sum(v ** 2 for v in vector2))
    return dot_product / (magnitude1 * magnitude2)

# Search images using Azure Search Index
def search_images_in_index(query):
    try:
        query_vector = get_text_embedding(query)
        if not query_vector:
            return None

        # Perform vector search, retrieve top 5 results
        results = search_client.search(
            search_text=None,
            vector_queries=[VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=5,
                fields="embedding"
            )]
        )
        
        # Extract image filenames
        top_images = [doc["filename"] for doc in results]
        return top_images if top_images else None
    except Exception as e:
        print(f"Error searching images for query '{query}': {e}")
        return None
    
    # Filter images based on the threshold
def filter_images_by_threshold(image_list, prompt):
    similarity_scores = []
    
    # Calculate similarity scores for all images
    for image_filename in image_list:
        image_path = os.path.join("static/images/", image_filename)
        
        try:
            image_embedding = get_image_embedding(image_path)
            
            if image_embedding:
                similarity_score = get_cosine_similarity(image_embedding, get_text_embedding(prompt))
                
                if similarity_score > THRESHOLD:
                    similarity_scores.append((image_path, similarity_score))
                    
        except Exception as e:
            print(f"Error processing {image_filename}: {e}")
    
    # Sort the images by similarity score in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # If there are no images above the threshold, return an empty list
    if not similarity_scores:
        return []
    
    # Get the highest similarity score (top image)
    top_similarity_score = similarity_scores[0][1]
    
    # Initialize the final filtered list with the top image
    final_filtered_images = [similarity_scores[0]]
    
    # Loop through the remaining images and apply the similarity difference check
    for image_path, score in similarity_scores[1:]:
        if top_similarity_score - score < DIFFERENCE_THRESHOLD:
            final_filtered_images.append((image_path, score))
    
    return final_filtered_images
