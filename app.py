import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
import pytesseract
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

app = Flask(__name__)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# Configure app to use the upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1):    
    # Calculate grayscale histogram
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

@app.route('/process', methods=['POST'])
def process():
    # Check if a file was uploaded
    if 'image' not in request.files:
        return 'No file uploaded.', 400

    file = request.files['image']

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Save the uploaded file to the upload folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        extension = filename.split('.')[1]

        if extension == 'pdf':
            doc = convert_from_path(file_path)
            path, fileName = os.path.split(file_path)
            fileBaseName, fileExtension = os.path.splitext(fileName)
            result = []

            for page_number, page_data in enumerate(doc):
                txt = pytesseract.image_to_string(page_data).encode("utf-8")
                result.append(txt)
                
            final_text = " ".join([str(text) for text in result])

            return render_template('result.html', processed_text = final_text)
        else:
            # Perform image processing
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            median_blur = cv2.medianBlur(gray, 3)
            processed_image, _, _ = automatic_brightness_and_contrast(median_blur)
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prcessed_image.jpg')
            cv2.imwrite(processed_path, processed_image)
            
            # Perform OCR to extract the text from the processed image
            processed_text = pytesseract.image_to_string(processed_image)

            return render_template('result.html', original_image=file_path, processed_image=processed_path, processed_text=processed_text)

    return 'Invalid file format.', 400

if __name__ == '__main__':
    app.run(debug=True)
