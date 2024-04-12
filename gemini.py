"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""

import google.generativeai as genai

genai.configure(api_key="AIzaSyApymR9uyb48LlI9z3CHbW54EYbCCMmad0")

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 1024,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ["hi"]
  },
  {
    "role": "model",
    "parts": ["Hello! 👋  How can I help you today?"]
  },
])

s = "i was coming from school and saw an accident"

convo.send_message(f"What kind of emotion is this text expressing, say it in no more than one word from these emotions (Happy, Angry, Surprise, Sad, Fear, and Neutral): {s}\nAI:")
text = convo.last.text
print(text)
convo.send_message(f"You are a counsellor and chat as a counsellor yourself and dont give heading with ** just act as counsellor.Using this input from the user: {s}, and the emotion of the user from that text: {text}, say something which will help their mode and give recommendations to make their situation better.ask related questions as a counsellor\nAI:")
print(convo.last.text)