## Getting started
1. Generate a chat_history folder
2. Generate a secret.json file. This file should contain the following:

{
  "project_id": "your_project_id",
  "location": "location for Gemini",
  "openai_api_key": "your_openai_api_key"
}

if you don't use openai, you can remove the openai_api_key from the secret.json file. 

3. pip3 install -r requirements.txt
   
4. Run the following command: 
python3 -m chat_bot

5. Open the browser and go to http://127.0.0.1:6091/ 
