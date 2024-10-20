from flask import Flask, render_template, request, jsonify
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import pickle
import os
import datetime
import re

app = Flask(__name__)

# Configure Vertex AI
PROJECT_ID = "acv-data"  # Replace with your Google Cloud project ID
LOCATION = "us-central1"  # Replace with your model's location
MODEL_NAME = "gemini-1.5-pro-002"  # Or another Gemini model name

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Create and start a chat session globally
model = GenerativeModel(model_name=MODEL_NAME, generation_config=GenerationConfig(temperature=0.2))
# chat_session = model.start_chat()  # This is now global, or per user session if required

chat_session = None  # Initialize chat_session as None

@app.route("/", methods=["GET"])  # Separate route for serving index.html
def index():
    global chat_session  # Access the global chat_session
    if chat_session is None:
        chat_session = model.start_chat() # Start a new chat session if one doesn't exist
    return render_template("index.html")


@app.route("/chat", methods=["POST"]) # Now exclusively for chat messages
def chat():
    global chat_session
    user_message = request.form.get("user_message")
    if not user_message:
        return jsonify({"error": "Message cannot be empty."})
    response = generate_gemini_response(user_message)
    return jsonify({"bot_message": response})

@app.route("/save_history", methods=["POST"])
def save_history():
    global chat_session
    today = datetime.date.today()
    formatted_date = today.strftime('%Y_%m_%d')
    input_string = chat_session._history[0].text
    words = input_string.split()  # Split the string into a list of words
    words = [re.sub(r'\W+', '', word) for word in words]
    if len(words) >= 5:
        file_name = '_'.join(words[:5])
    else:
        file_name = '_'.join(words)
    file_name = "chat_history/" + file_name + "_" + formatted_date + ".pkl"
    try:
        with open(file_name, "wb") as file:
            pickle.dump(chat_session._history, file)
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error saving chat history: {e}")
        return jsonify({"status": "error"}), 500

@app.route("/load_history", methods=["POST"])
def load_history():
    if os.path.exists("chat_history.pkl"):
        try:
            with open("chat_history.pkl", "rb") as file:
                load_history = pickle.load(file)
                chat_session._history = load_history
            return jsonify({"status": "success"})  #jsonify({"chat_history": chat_session._history})
        except Exception as e:
            print(f"Error loading chat history: {e}")
            return jsonify({"status": "error"}), 500
    else:
        return jsonify({"chat_history": []})

@app.route("/save_best_answer", methods=["POST"])
def save_best_answer():
    global chat_session
    data = request.get_json()
    chat_history_dict = [entry.to_dict() for entry in chat_session._history]
    # user_messages = data['user_messages']
    # print(data['user_messages'])
    best_answer = data['best_answer']
    try:
        with open("best_answer_history.pkl", "wb") as file:
            pickle.dump({"chat_history": chat_history_dict, "best_answer": best_answer}, file)
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error saving best answer: {e}")
        return jsonify({"status": "error"}), 500


def generate_gemini_response(user_message):
    try:
        # Send the new user message to the ongoing chat session
        response = chat_session.send_message(user_message)
        # Extract the response content
        if isinstance(response, str):
            bot_message = response
        else:
            bot_message = response.text if hasattr(response, 'text') else str(response)

        return bot_message
    except Exception as e:
        print(f"Error generating Gemini response: {e}")
        return "Error: Could not generate a response."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6091, debug=True)
