from flask import Flask, request, jsonify
import util

app = Flask(__name__)

# Server supporting simple API called 'Hello'
# Might have to manually type in 'localhost:8000/hello' to access server 
@app.route('/classify_image/', methods=['GET','POST'])
def classify_image(): # This function will do image classifying using saved model 
    image_data = request.form['image_data'] # B64 encoded string
    
    response = jsonify(util.classify_image(image_data)) # Convert into JSON
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response
    

if __name__ == "__main__":
    print("Starting Python Flask Server for NBA Athlete Image Classification")
    util.load_saved_artifacts()
    app.run(port=8000) # Running server on port 5500
