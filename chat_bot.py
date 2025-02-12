import datetime
import json
import pickle
import re

import vertexai
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from vertexai.generative_models import GenerationConfig, GenerativeModel

app = Flask(__name__)

# Import secrets for Vertex AI and OpenAI
secret = json.load(open("secret.json"))

# Configure Vertex AI
PROJECT_ID = secret["project_id"]
LOCATION = secret["location"]
MODEL_NAME = "gemini-2.0-pro-exp-02-05"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# OpenAI GPT API key
OPENAI_API_KEY = secret["openai_api_key"]
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize global variables
chat_session = None  # For Gemini
current_model = None  # Tracks active LLM
gpt_chat_history = []  # For GPT chat history


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/new_session", methods=["POST"])
def new_session():
    global chat_session, current_model, gpt_chat_history

    # Get the selected LLM from the client
    selected_model = request.json.get("llm")
    if not selected_model:
        return jsonify({"error": "No LLM selected"}), 400

    # Initialize the selected LLM
    if selected_model == "Gemini":
        model = GenerativeModel(
            model_name=MODEL_NAME, generation_config=GenerationConfig(temperature=0.2)
        )
        chat_session = model.start_chat()
        current_model = "Gemini"
    elif selected_model == "GPT":
        chat_session = None
        gpt_chat_history = []  # Clear history for a new GPT session
        current_model = "GPT"
    else:
        return jsonify({"error": "Invalid LLM selected"}), 400

    return jsonify({"message": f"New session started with {selected_model}"})


@app.route("/chat", methods=["POST"])
def chat():
    global chat_session, current_model

    user_message = request.form.get("user_message")
    if not user_message:
        return jsonify({"error": "Message cannot be empty."})

    try:
        if current_model == "Gemini":
            response = generate_gemini_response(user_message)
        elif current_model == "GPT":
            response = generate_gpt_response(user_message)
        else:
            return jsonify({"error": "No LLM selected for the session."}), 400

        return jsonify({"bot_message": response})
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route("/save_history", methods=["POST"])
def save_history():
    global chat_session, current_model

    today = datetime.date.today()
    formatted_date = today.strftime("%Y_%m_%d")

    if current_model == "Gemini":
        history_data = chat_session._history
        input_string = history_data[0].text if history_data else "no_data"
    elif current_model == "GPT":
        history_data = gpt_chat_history
        input_string = history_data[0]["content"] if history_data else "no_data"
    else:
        return jsonify({"error": "No active session"}), 400

    words = re.sub(r"\W+", " ", input_string).split()
    file_name = (
        "chat_history/" + "_".join(words[:5])
        if len(words) >= 5
        else "_".join(words) + f"_{current_model.lower()}_{formatted_date}.pkl"
    )

    try:
        with open(file_name, "wb") as file:
            pickle.dump(history_data, file)
        return jsonify({"status": "success", "file": file_name})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/load_history", methods=["POST"])
def load_history():
    global chat_session, gpt_chat_history, current_model

    if "history_file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files["history_file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400

    try:
        loaded_history = pickle.load(file)

        if current_model == "Gemini":
            chat_session._history = loaded_history
            history_content = [
                {"role": entry.role, "content": part.text}
                for entry in loaded_history
                for part in entry.parts
            ]
        elif current_model == "GPT":
            gpt_chat_history = loaded_history
            history_content = gpt_chat_history
        else:
            return jsonify({"status": "error", "message": "No active session"}), 400

        return jsonify({"status": "success", "history": history_content})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def generate_gemini_response(user_message):
    try:
        response = chat_session.send_message(user_message)
        return response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        print(f"Error generating Gemini response: {e}")
        return f"Error: {str(e)}"


def generate_gpt_response(user_message):
    global gpt_chat_history

    try:
        gpt_chat_history.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model="gpt-4o", messages=gpt_chat_history
        )
        print(response)
        bot_message = response.choices[0].message.content
        gpt_chat_history.append({"role": "assistant", "content": bot_message})
        return bot_message
    except Exception as e:
        print(f"Error generating GPT response: {e}")  # Log the error details
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6091, debug=True)
