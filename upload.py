import requests

# Specify the URL of the upload endpoint
upload_url = "http://localhost:8000/upload"

# Path to the PDF file we want to upload
pdf_path = "file:///C:/Users/Aravi/Downloads/Ai%20Use%20in%20Schools.pdf"

# Send a POST request with the PDF file
with open(pdf_path, "rb") as file:
    response = requests.post(upload_url, files={"file": file})

# Check the response status code
if response.status_code == 200:
    print("PDF file uploaded successfully!")
else:
    print("Failed to upload the PDF file.")
