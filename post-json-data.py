import requests

api_url = "https://jsonplaceholder.typicode.com/posts"

payload = {
    "userId": 1,
    "title": "Posting JSON data using Python",
    "body": "This is a sample POST request sending JSON data to an API endpoint."
}

response = requests.post(api_url, json=payload)

print(f"Status Code: {response.status_code}")
print("Response JSON:", response.json())
