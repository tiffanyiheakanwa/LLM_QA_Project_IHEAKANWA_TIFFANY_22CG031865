"""
LLM Q&A CLI Application - Using Google Gemini
A command-line interface for asking questions using Gemini API
"""

import os
import re
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API key missing!")
    exit()


class LLMQuestionAnswering:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("API key not found. Set GEMINI_API_KEY environment variable")
        
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
        self.model = "gemini-2.5-flash"
    
    def preprocess_question(self, question: str) -> str:
        processed = question.lower().strip()
        processed = ' '.join(processed.split())
        processed = re.sub(r'[^\w\s\?.,!+\-*/]', '', processed)
        print(f"\n[Preprocessed Question]: {processed}")
        return processed
    
    def construct_prompt(self, question: str) -> str:
        return f"""You are a helpful AI assistant. Please answer the following question clearly and concisely.

Question: {question}

Answer:"""
    
    def query_llm(self, question: str) -> dict:
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": self.construct_prompt(question)
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        try:
            print("\n[Sending request to Gemini API...]")
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying Gemini: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return {"error": str(e)}
    
    def extract_answer(self, response: dict) -> str:
        if "error" in response:
            return f"Error: {response['error']}"
        
        try:
            candidates = response.get('candidates', [])
            if candidates:
                parts = candidates[0].get('content', {}).get('parts', [])
                if parts:
                    return parts[0].get('text', '').strip()
        except:
            return "Unable to parse response from Gemini"
        
        return "Unable to get response from Gemini"
    
    def ask_question(self, question: str) -> str:
        print("\n" + "="*60)
        print("PROCESSING YOUR QUESTION (GEMINI)")
        print("="*60)
        
        processed_question = self.preprocess_question(question)
        response = self.query_llm(processed_question)
        answer = self.extract_answer(response)
        return answer


def main():
    print("="*60)
    print("LLM QUESTION & ANSWERING SYSTEM")
    print("Powered by Google Gemini")
    print("="*60)
    print("\nWelcome! Ask me anything, or type 'quit' to exit.\n")
    
    try:
        qa_system = LLMQuestionAnswering()
        print("✓ Successfully connected to Gemini API\n")
    except ValueError as e:
        print(f"✗ Error: {e}")
        return
    
    while True:
        try:
            question = input("\n💬 Your Question: ").strip()
            
            if not question:
                print("Please enter a question.")
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Q&A system. Goodbye!")
                break
            
            answer = qa_system.ask_question(question)
            
            print("\n" + "="*60)
            print("🤖 ANSWER:")
            print("="*60)
            print(f"{answer}\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting... Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
