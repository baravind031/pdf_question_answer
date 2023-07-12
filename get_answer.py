import requests

# Specify the URL of the getanswer endpoint
get_answer_url = "http://localhost:8000/getanswer"

# Send a GET request to retrieve the answer
response = requests.get(get_answer_url)

# Get the response JSON
response_json = response.json()

# Check if the answer is available
if "answer" in response_json:
    answer = response_json["answer"]
    print(f"Answer: {answer}")
else:
    print("No answer available.")
