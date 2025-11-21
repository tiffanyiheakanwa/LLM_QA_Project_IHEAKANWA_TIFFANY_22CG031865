"""
Flask Web Application for LLM Q&A System
Using Google Gemini API
"""

from flask import Flask, render_template, request, jsonify
import os
import re
import requests
from datetime import datetime

app = Flask(__name__)

class LLMQuestionAnswering:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("API key not found")
        
        # CORRECT MODEL URL
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
        self.model = "gemini-2.5-flash"
    
    def preprocess_question(self, question):
        processed = question.lower().strip()
        processed = ' '.join(processed.split())
        processed = re.sub(r'[^\w\s\?.,!-]', '', processed)
        return processed
    
    def construct_prompt(self, question):
        return f"""You are a helpful AI assistant. Please answer the following question clearly and concisely.

Question: {question}

Answer:"""
    
    def query_llm(self, question):
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [{
                "parts": [{"text": self.construct_prompt(question)}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def extract_answer(self, response):
        if "error" in response:
            return f"Error: {response['error']}"
        
        try:
            # Gemini returns: candidates → content → parts → text
            candidates = response.get('candidates', [])
            if candidates:
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if parts:
                    return parts[0].get('text', '').strip()
        except:
            return "Unable to parse response from Gemini"
        
        return "Unable to get a valid response from Gemini"
    
    def ask_question(self, question):
        processed_question = self.preprocess_question(question)
        response = self.query_llm(processed_question)
        answer = self.extract_answer(response)
        return processed_question, answer


# Initialize system
try:
    qa_system = LLMQuestionAnswering()
    print("✓ Successfully initialized Gemini API")
except ValueError:
    qa_system = None
    print("✗ Gemini API key not found")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    if not qa_system:
        return jsonify({'error': 'API key not configured.'}), 500
    
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Please enter a question'}), 400
    
    try:
        processed_question, answer = qa_system.ask_question(question)
        
        return jsonify({
            'original_question': question,
            'processed_question': processed_question,
            'answer': answer,
            'model': 'Gemini Turbo',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model': 'Gemini Turbo',
        'api_configured': qa_system is not None
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("LLM Q&A System - Powered by Google Gemini")
    print("="*60)
    print(f"Status: {'✓ Ready' if qa_system else '✗ Not configured'}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
