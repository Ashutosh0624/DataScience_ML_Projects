# organising the Image differentiator code in a reusable funtion

from skimage.metrics import structural_similarity as ssim
import imutils
import cv2
from PIL import Image

def detect_fake(original_image_path, tampered_image_path):
    """
        Detects tampering between two images using Structural Similarity Index (SSIM).

        Args:
            original_image_path (str): Path to the original image.
            tampered_image_path (str): Path to the tampered image.

        Returns:
            tuple: SSIM score, contour-highlighted tampered image, difference image, and threshold image.
    """
    # load the image using openCv
    original = cv2.imread(original_image_path)
    tampered = cv2.imread(tampered_image_path)

    # resize the images to ensure uniformity
    original = cv2.resize(original, (250, 260))
    tampered = cv2.resize(tampered, (250, 260))

    # convert the images to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

    # compute the structural similarity Index (SSIM) and difference image
    (score, diff) = ssim(original_gray, tampered_gray, full=True)
    diff = (diff * 255).astype("uint8")

    # Apply thresholding to the difference image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours of the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the countours to draw the bounding boxes
    for c in cnts:
        #compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x+w, y+h), (0,255, 0), 2)
        cv2.rectangle(tampered, (x, y), (x+w, y+h), (0,255, 0), 2)

    return score, tampered, diff, thresh

'''
Explanation of the Function
Inputs:

original_image_path: Path to the original PAN Card image.
tampered_image_path: Path to the tampered PAN Card image.
Workflow:

Load the images using OpenCV.
Resize images to ensure they are of the same size.
Convert both images to grayscale.
Compute the Structural Similarity Index (SSIM) to measure the similarity between the two images and generate a difference image.
Apply thresholding to the difference image to highlight significant changes.
Find contours in the thresholded image and draw bounding boxes on the tampered image to mark differences.
Outputs:

score: SSIM score (a value closer to 1 indicates more similarity).
tampered: The tampered image with bounding boxes drawn around detected differences.
diff: The difference image highlighting pixel-wise variations.
thresh: The thresholded binary image.
'''