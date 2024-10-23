import os
import requests
from flask import Flask, render_template, request, jsonify
from waitress import serve
from flask_cors import CORS
from image_search_app import *

app = Flask(__name__)

cors = CORS(app)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_query = request.form['message']
        
        # Step 1: Get top 5 images from Azure Search Index
        top_images = search_images_in_index(user_query)
        
        if top_images:
            # Step 2: Filter images based on the threshold
            filtered_images_with_scores = filter_images_by_threshold(top_images, user_query)
            
            if filtered_images_with_scores:
                images = [{'path': img_path, 'score': score} for img_path, score in filtered_images_with_scores]
                return jsonify({'images': images, 'message': 'Here are my best suggestions from left to right.'})
        
        return jsonify({'message': 'No relevant images found. Enter a valid query.'})
    except Exception as e:
        print(f"Error handling user query: {e}")
        return jsonify({'message': 'An error occurred while processing your request.'})


if __name__ == '__main__':
    app.run(debug=True)
    #serve(app, host="127.0.0.1", port = 8081)
