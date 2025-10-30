# Importing the 'requests' library to make HTTP requests
import requests

# The URL of the fake online REST API that provides sample user data
url = "https://jsonplaceholder.typicode.com/users"

# Sending a GET request to the URL and storing the response
response = requests.get(url)

# Checking if the request was successful (HTTP status code 200 means OK)
if response.status_code == 200:
    # Converting the JSON response to a Python dictionary/list
    data = response.json()
    
    # Loop through only the first three users from the data
    for user in data[:3]:
        # Printing each user's name
        print("Name:", user["name"])
        # Printing each user's email
        print("Email:", user["email"])
        # Printing an empty line for better readability
        print()

# If the request failed, print the error status code
else:
    print("Error: ", response.status_code)
