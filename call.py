import requests

API_URL = "http://127.0.0.1:8000"  # Change this to your actual API URL

# Call the /whiteboard endpoint
def call_whiteboard(question, latex_text, command, chat_history):
    url = f"{API_URL}/whiteboard"
    payload = {
        "question": question,
        "whiteboard_content_latex_text": latex_text,
        "command": command,
        "chat_history": chat_history
    }
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
if __name__ == "__main__":
    chat_history = ""
    
    whiteboard_response = call_whiteboard(
        "Solve for x in the equation x^2 - 4 = 0",
        "x^2 - 4 = 0",
        "What should I do next",
        chat_history
    )
    print("Whiteboard Response:", whiteboard_response)
