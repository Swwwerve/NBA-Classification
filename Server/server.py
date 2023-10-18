from flask import Flask, request, jsonify

app = Flask(__name__)

# Server supporting simple API called 'Hello'
@app.route('/hello')
def hello():
    return 'hi'

if __name__ == "__main__":
    # Running server on port 5000
    app.run(port=5500)