from flask import Flask, request, jsonify
import cv2
import numpy as np
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

@app.route('/save_faceimage', methods=['POST'])
def save():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({'status': 'fail', 'message': 'Image or name not provided'}), 400
    
    file = request.files['image']
    user_name = request.form['name']

    if not file or not user_name:
        return jsonify({'status': 'fail', 'message': 'Invalid image data or name'}), 400
    
    try:
        # Read the image in OpenCV format
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'status': 'fail', 'message': 'Invalid image data'}), 400
        
        new_image_path = f'{user_name}.jpg'
        cv2.imwrite(new_image_path, img)
        return jsonify({'status': 'success', 'message': f'Image saved as {new_image_path}'})
    
    except Exception as e:
        app.logger.error(f'Error processing image: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)
