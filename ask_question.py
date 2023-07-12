import requests

# Specify the URL of the askquestion endpoint
ask_question_url = "http://localhost:8000/askquestion"

# The question we want to ask
question = "what is the use of Early Warning Systems"

# Send a POST request with the question
response = requests.post(ask_question_url, json={"question": question})

# Get the response JSON
response_json = response.json()

# Check if the question was answered successfully
if "answer" in response_json:
    answer = response_json["answer"]
    print(f"Answer: {answer}")
else:
    message = response_json["message"]
    print(f"Failed to get the answer: {message}")
