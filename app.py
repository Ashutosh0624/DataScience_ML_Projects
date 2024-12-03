from flask import Flask, render_template, request
import os
import cv2  # Make sure to import OpenCV
from imagedifferentiator import detect_fake  # Import your detect_fake function

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get uploaded images
        image1 = request.files['image1']
        image2 = request.files['image2']

        # Save the images to the upload folder
        image1_path = os.path.join(app.config['UPLOAD_FOLDER'], image1.filename)
        image2_path = os.path.join(app.config['UPLOAD_FOLDER'], image2.filename)
        image1.save(image1_path)
        image2.save(image2_path)

        # Call the detection function
        score, tampered_image, diff_image, thresh_image = detect_fake(image1_path, image2_path)

        # Save the tampered image with bounding boxes
        tampered_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "tampered_output.jpg")
        cv2.imwrite(tampered_image_path, tampered_image)

        # Pass the score and tampered image to the template
        return render_template("index.html", score=score, result_image="uploads/tampered_output.jpg")

    # For GET requests, render the upload form
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
