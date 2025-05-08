import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
import logging
import cv2
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path of the directory this script is in
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Flask app with explicit template folder
app = Flask(__name__, 
            template_folder=os.path.join(SCRIPT_DIR, 'templates'),
            static_folder=os.path.join(SCRIPT_DIR, 'static'))

app.config['UPLOAD_FOLDER'] = os.path.join(SCRIPT_DIR, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create necessary folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, 'templates'), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, 'static'), exist_ok=True)

# Class labels for the model
CLASS_LABELS = {
    0: 'Bacterial Pneumonia',
    1: 'Covid-19',
    2: 'Myocardial Infarction (ECG)',
    3: 'History of MI (ECG)',
    4: 'Abnormal Heartbeat (ECG)',
    5: 'Normal ECG',
    6: 'Normal X-ray',
    7: 'Viral Pneumonia'
}

# Load the TensorFlow model
try:
    model_path = os.path.join(SCRIPT_DIR, "my_mobilenet_model.keras")
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        # Try the original path as fallback
        model = tf.keras.models.load_model("C:/Users/pavan/Downloads/my_mobilenet_model.keras")
    logger.info("Model loaded successfully ✅")
    print("✅ Model loaded successfully. Server starting...")
    print("Access the application at http://127.0.0.1:5000/")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    print(f"❌ Error loading model: {e}")

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess the image for the model"""
    try:
        # Read image
        img = cv2.imread(image_path)
        # Resize to the input size expected by your model
        img = cv2.resize(img, (224, 224))  # Adjust size as per your model requirements
        # Normalize pixel values
        img = img / 255.0
        # Expand dimensions to match model input shape [batch_size, height, width, channels]
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

# Add a function to create index.html file if it doesn't exist
def create_index_template():
    """Create the index.html template file if it doesn't exist"""
    template_path = os.path.join(SCRIPT_DIR, 'templates', 'index.html')
    
    if not os.path.exists(template_path):
        logger.info("Creating index.html template file")
        
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Image Analyzer</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .results-container {
            display: none;
            margin-top: 20px;
        }
        .spinner-border {
            display: none;
        }
        .prediction-bar {
            height: 24px;
            transition: width 0.6s ease;
        }
        .upload-container {
            border: 2px dashed #ccc;
            border-radius: 5px;
            padding: 25px;
            text-align: center;
            cursor: pointer;
        }
        .upload-container:hover {
            border-color: #0d6efd;
        }
        #preview-image {
            max-height: 300px;
            margin: 15px auto;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card shadow">
                    <div class="card-header bg-primary text-white">
                        <h3 class="mb-0">Medical Image Analyzer</h3>
                    </div>
                    <div class="card-body">
                        <p class="lead mb-4">Upload an X-ray or ECG image for analysis</p>
                        
                        <form id="upload-form" enctype="multipart/form-data">
                            <div class="upload-container mb-3" id="drop-area">
                                <div>
                                    <img src="/static/upload-icon.png" alt="Upload" width="48" height="48" onerror="this.src='/api/placeholder/48/48'; this.onerror=null;">
                                    <p>Drag & drop an image here or click to browse</p>
                                    <input type="file" id="file-input" name="file" accept=".jpg,.jpeg,.png" class="form-control" style="display: none;">
                                    <button type="button" id="browse-button" class="btn btn-outline-primary">Browse Files</button>
                                </div>
                            </div>
                            
                            <div class="text-center">
                                <img id="preview-image" class="img-fluid d-none" alt="Preview">
                                <div class="spinner-border text-primary mt-3" role="status" id="loading-spinner">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                                <button type="submit" id="analyze-button" class="btn btn-primary mt-3" disabled>Analyze Image</button>
                            </div>
                        </form>
                        
                        <div id="results-container" class="results-container mt-4">
                            <div class="card">
                                <div class="card-header bg-light">
                                    <h5 class="mb-0">Analysis Results</h5>
                                </div>
                                <div class="card-body">
                                    <h4 id="result-diagnosis" class="mb-3"></h4>
                                    <div id="image-type-container" class="mb-2">
                                        <span class="fw-bold">Image Type:</span>
                                        <span id="image-type"></span>
                                    </div>
                                    <div class="mb-2">
                                        <span class="fw-bold">Confidence:</span>
                                        <span id="confidence-score"></span>
                                    </div>
                                    
                                    <h5 class="mt-4">All Probabilities</h5>
                                    <div id="probability-bars"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const dropArea = document.getElementById('drop-area');
            const fileInput = document.getElementById('file-input');
            const browseButton = document.getElementById('browse-button');
            const previewImage = document.getElementById('preview-image');
            const analyzeButton = document.getElementById('analyze-button');
            const uploadForm = document.getElementById('upload-form');
            const loadingSpinner = document.getElementById('loading-spinner');
            const resultsContainer = document.getElementById('results-container');
            
            // Handle browse button click
            browseButton.addEventListener('click', () => {
                fileInput.click();
            });
            
            // Handle drag events
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, unhighlight, false);
            });
            
            function highlight() {
                dropArea.classList.add('border-primary');
            }
            
            function unhighlight() {
                dropArea.classList.remove('border-primary');
            }
            
            // Handle dropped files
            dropArea.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                handleFiles(files);
            }
            
            // Handle file selection
            fileInput.addEventListener('change', function() {
                handleFiles(this.files);
            });
            
            function handleFiles(files) {
                if (files.length > 0) {
                    const file = files[0];
                    
                    // Check if file is an allowed image type
                    if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
                        alert('Please upload a valid image file (JPG or PNG)');
                        return;
                    }
                    
                    // Show preview
                    previewImage.classList.remove('d-none');
                    previewImage.src = URL.createObjectURL(file);
                    
                    // Enable analyze button
                    analyzeButton.disabled = false;
                }
            }
            
            // Handle form submission
            uploadForm.addEventListener('submit', function(e) {
                e.preventDefault();
                
                if (fileInput.files.length === 0) {
                    alert('Please select an image first');
                    return;
                }
                
                // Show loading spinner
                loadingSpinner.style.display = 'inline-block';
                analyzeButton.disabled = true;
                resultsContainer.style.display = 'none';
                
                // Create FormData and send request
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                fetch('https://lung-detection.onrender.com/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading spinner
                    loadingSpinner.style.display = 'none';
                    analyzeButton.disabled = false;
                    
                    if (data.error) {
                        alert('Error: ' + data.error);
                        return;
                    }
                    
                    // Display results
                    document.getElementById('result-diagnosis').textContent = data.class;
                    document.getElementById('image-type').textContent = data.image_type;
                    document.getElementById('confidence-score').textContent = 
                        (data.confidence * 100).toFixed(2) + '%';
                    
                    // Display probability bars
                    const probabilityBarsContainer = document.getElementById('probability-bars');
                    probabilityBarsContainer.innerHTML = '';
                    
                    // Sort probabilities in descending order
                    const sortedProbabilities = Object.entries(data.all_probabilities)
                        .sort((a, b) => b[1] - a[1]);
                    
                    for (const [className, probability] of sortedProbabilities) {
                        const percentage = (probability * 100).toFixed(2);
                        const barClass = className === data.class ? 'bg-success' : 'bg-primary';
                        
                        const barHtml = `
                            <div class="mb-2">
                                <div class="d-flex justify-content-between mb-1">
                                    <span>${className}</span>
                                    <span>${percentage}%</span>
                                </div>
                                <div class="progress" style="height: 20px;">
                                    <div class="progress-bar ${barClass} prediction-bar" 
                                         role="progressbar" 
                                         style="width: ${percentage}%" 
                                         aria-valuenow="${percentage}" 
                                         aria-valuemin="0" 
                                         aria-valuemax="100"></div>
                                </div>
                            </div>
                        `;
                        
                        probabilityBarsContainer.innerHTML += barHtml;
                    }
                    
                    // Show results container
                    resultsContainer.style.display = 'block';
                })
                .catch(error => {
                    loadingSpinner.style.display = 'none';
                    analyzeButton.disabled = false;
                    alert('Error: ' + error.message);
                });
            });
        });
    </script>
</body>
</html>
"""
        
        # Ensure the templates directory exists
        os.makedirs(os.path.join(SCRIPT_DIR, 'templates'), exist_ok=True)
        
        # Write the HTML content to the file
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Created index.html template at {template_path}")

# Call function to create template
create_index_template()

@app.route('/')
def home():
    """Render the home page"""
    # Check if template exists before rendering
    template_path = os.path.join(SCRIPT_DIR, 'templates', 'index.html')
    if not os.path.exists(template_path):
        create_index_template()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image uploads and return predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image
        processed_img = preprocess_image(file_path)
        
        if processed_img is None:
            return jsonify({'error': 'Error processing image'})
        
        # Make prediction
        if model is not None:
            try:
                prediction = model.predict(processed_img)
                
                # Get the predicted class index and probability
                pred_class_index = np.argmax(prediction[0])
                pred_probability = float(prediction[0][pred_class_index])
                
                # Get class name from our mapping
                pred_class_name = CLASS_LABELS.get(pred_class_index, f"Unknown Class ({pred_class_index})")
                
                # Determine if it's ECG or X-ray type
                if pred_class_index in [2, 3, 4, 5]:
                    image_type = "ECG"
                else:
                    image_type = "X-ray"
                
                result = {
                    'prediction': pred_probability,
                    'class': pred_class_name,
                    'class_index': int(pred_class_index),
                    'confidence': pred_probability,
                    'image_path': os.path.join('static', 'uploads', filename),
                    'image_type': image_type,
                    'all_probabilities': {CLASS_LABELS[i]: float(prediction[0][i]) for i in range(len(prediction[0]))}
                }
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error making prediction: {e}")
                return jsonify({'error': f'Error making prediction: {str(e)}'})
        else:
            return jsonify({'error': 'Model not loaded'})
    
    return jsonify({'error': 'File type not allowed'})

# Add a debugging route to check paths
@app.route('/debug')
def debug():
    paths = {
        'SCRIPT_DIR': SCRIPT_DIR,
        'template_folder': os.path.join(SCRIPT_DIR, 'templates'),
        'static_folder': os.path.join(SCRIPT_DIR, 'static'),
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'template_exists': os.path.exists(os.path.join(SCRIPT_DIR, 'templates', 'index.html')),
        'templates_dir_exists': os.path.exists(os.path.join(SCRIPT_DIR, 'templates')),
        'working_dir': os.getcwd(),
        'class_labels': CLASS_LABELS
    }
    return jsonify(paths)

if __name__ == '__main__':
    app.run(debug=True)