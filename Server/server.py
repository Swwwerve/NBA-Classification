from flask import Flask, request, jsonify

app = Flask(__name__)

# Server supporting simple API called 'Hello'
# Might have to manually type in 'localhost:8000/hello' to access server 
@app.route('/hello/')
def hello():
    return 'hi'

if __name__ == "__main__":
    # Running server on port 5500
    app.run(port=8000)