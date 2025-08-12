import os
import pickle
import re
from dotenv import load_dotenv
import numpy as np
from flask import Flask, render_template, request, jsonify
from google import genai
import os
# Using PyTorch-based Hugging Face T5
from transformers import T5Tokenizer, T5ForConditionalGeneration
load_dotenv()

GEMINI_API_KEY = os.getenv("API_KEY")  # Gemini API Key for Therapist Chatbot
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = Flask(__name__)

# 1) Load Disorder Model

disorder_model_path = os.path.join("models", "disorder_model.pkl")
with open(disorder_model_path, "rb") as f:
    disorder_model = pickle.load(f)

# 2) Load Emotion Model
emotion_model_path = os.path.join("models", "emotion_model.pkl")
with open(emotion_model_path, "rb") as f:
    emotion_model_data = pickle.load(f)
    emotion_model = emotion_model_data['model']
    emotion_vectorizer = emotion_model_data['vectorizer']

# 3) Load Summarization Model (PyTorch format from `my_model_dir`)

summarization_model_dir = "my_model_dir"

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(summarization_model_dir)
summarization_model = T5ForConditionalGeneration.from_pretrained(summarization_model_dir)

# Function to summarize text
def summarize_text(input_text):
    # Encode the input text
    input_ids = tokenizer.encode(
        "summarize: " + input_text,
        return_tensors="pt",     # "pt" = PyTorch
        max_length=512,
        truncation=True
    )
    # Generate summary
    summary_ids = summarization_model.generate(
        input_ids,
        max_length=150,
        num_beams=5,
        early_stopping=True
    )
    # Decode and return
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Helper: Clean text for emotion model
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Route: Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Route: Predict Disorder (POST)
@app.route('/predict_disorder', methods=['POST'])
def predict_disorder():
    disorder_mapping = {
        0: 'ADHD', 1: 'ASD', 2: 'Loneliness', 3: 'MDD', 4: 'OCD', 5: 'PDD',
        6: 'PTSD', 7: 'Anxiety', 8: 'Bipolar', 9: 'Eating Disorder',
        10: 'Psychotic Depression', 11: 'Sleeping Disorder'
    }

    # Make sure these match the exact feature order expected by your model
    feature_names = [
        'age', 'feeling.nervous', 'panic', 'breathing.rapidly', 'sweating',
        'trouble.in.concentration', 'having.trouble.in.sleeping',
        'having.trouble.with.work', 'hopelessness', 'anger', 'over.react',
        'change.in.eating', 'suicidal.thought', 'feeling.tired', 'close.friend',
        'social.media.addiction', 'weight.gain', 'introvert',
        'popping.up.stressful.memory', 'having.nightmares',
        'avoids.people.or.activities', 'feeling.negative',
        'trouble.concentrating', 'blaming.yourself', 'hallucinations',
        'repetitive.behaviour', 'seasonally', 'increased.energy'
    ]

    try:
        features = [float(request.form[feature]) for feature in feature_names]
    except KeyError as e:
        return render_template('index.html', prediction_text=f"Missing field: {e}")

    # Predict disorder
    prediction = disorder_model.predict([np.array(features)])
    disorder_code = prediction[0]
    disorder_name = disorder_mapping.get(disorder_code, "Unknown Disorder")

    return render_template(
        'disease_prediction.html',
        prediction_text=f'Predicted Disorder: {disorder_code} ({disorder_name})'
    )

# Route: Predict Emotion (POST via JSON fetch)
@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    text = request.json.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    cleaned_text = clean_text(text)
    text_vec = emotion_vectorizer.transform([cleaned_text])
    probabilities = emotion_model.predict_proba(text_vec)[0]
    emotion_index = probabilities.argmax()

    return jsonify({
        "emotion": emotion_model.classes_[emotion_index],
        "confidence": round(float(probabilities[emotion_index]), 2)
    })

# Route: Text Summarizer (GET/POST)
#   - On POST: Summarize user text
#   - Then feed the summary to emotion model
#   - Show both summary & predicted emotion
@app.route('/text_summarizer', methods=['GET', 'POST'])
def text_summarizer():
    summary_result = None
    emotion_result = None
    emotion_confidence = None

    if request.method == 'POST':
        user_text = request.form.get('summary_input', '')
        if user_text.strip():
            # 1) Summarize using the T5 model
            summary_result = summarize_text(user_text)

            # 2) Predict Emotion of the summary
            cleaned_summary = clean_text(summary_result)
            text_vec = emotion_vectorizer.transform([cleaned_summary])
            probabilities = emotion_model.predict_proba(text_vec)[0]
            emotion_index = probabilities.argmax()
            emotion_result = emotion_model.classes_[emotion_index]
            emotion_confidence = round(float(probabilities[emotion_index]), 2)

    return render_template(
        'text_summarizer.html',
        summary_result=summary_result,
        emotion_result=emotion_result,
        emotion_confidence=emotion_confidence
    )



@app.route('/emotion')
def emotion_page():
    return render_template('emotion_prediction.html')

@app.route('/summarizer')
def summarizer_page():
    return render_template('text_summarizer.html')

@app.route('/disease')
def disease_page():
    return render_template('disease_prediction.html')

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

# Therapist system prompt
THERAPIST_PROMPT = """
You are Dr. AI, a compassionate and professional AI therapist. Your role is to:

1. Listen actively and empathetically to the user's concerns
2. Ask thoughtful, open-ended questions to help users explore their feelings
3. Provide supportive and non-judgmental responses
4. Use therapeutic techniques like reflection, validation, and gentle guidance
5. Maintain professional boundaries while being warm and caring
6. Keep responses concise but meaningful (2-3 sentences max)
7. Never provide medical diagnoses or prescribe medications
8. Encourage professional help when appropriate
9. You are a helpful therapist. Respond to the user's messages with empathy and understanding. If you don't know the answer, say 'I'm not sure, but I can help you find resources.
Remember: You are here to support, listen, and guide - not to diagnose or treat medical conditions.

Respond as a therapist would in a real session.
"""

# Add this route to your app.py
@app.route('/chat_with_therapist', methods=['POST'])
def chat_with_therapist():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        conversation_history = data.get('history', [])
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Build conversation context
        conversation_context = THERAPIST_PROMPT + "\n\nConversation History:\n"
        
        # Add previous messages for context (limit to last 10 exchanges)
        recent_history = conversation_history[-20:] if len(conversation_history) > 20 else conversation_history
        for msg in recent_history:
            conversation_context += f"User: {msg}\n"
        
        # Add current message
        conversation_context += f"User: {user_message}\n\nTherapist:"
        
        # Generate response using Gemini
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=conversation_context,
        )
        
        therapist_response = response.text.strip()
        
        return jsonify({
            "response": therapist_response,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error in chat_with_therapist: {str(e)}")
        return jsonify({
            "error": "Sorry, I'm having trouble responding right now. Please try again.",
            "status": "error"
        }), 500



# Add this route to your existing app.py file

@app.route('/process_chat_session', methods=['POST'])
def process_chat_session():
    try:
        data = request.get_json()
        user_messages = data.get('user_messages', [])
        
        if not user_messages:
            return jsonify({"error": "No messages to process"}), 400
        
        # Combine all user messages into one text
        combined_text = ' '.join(user_messages)
        
        # 1) Summarize the combined user messages using T5 model
        summary = summarize_text(combined_text)
        
        # 2) Predict emotion of the summary
        cleaned_summary = clean_text(summary)
        text_vec = emotion_vectorizer.transform([cleaned_summary])
        probabilities = emotion_model.predict_proba(text_vec)[0]
        emotion_index = probabilities.argmax()
        emotion = emotion_model.classes_[emotion_index]
        confidence = float(probabilities[emotion_index])
        
        return jsonify({
            "summary": summary,
            "emotion": emotion,
            "confidence": confidence
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500






# Run the Flask App
if __name__ == "__main__":
    app.run(debug=True, port=5000)
