import cv2
import numpy as np

# Load parking image (replace with your image path)
image = cv2.imread("parking.jpg")

# Resize for better processing
image = cv2.resize(image, (800, 600))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply blur
blur = cv2.GaussianBlur(gray, (5, 5), 1)

# Edge detection
edges = cv2.Canny(blur, 50, 150)

# Threshold
_, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours (parking spaces)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

empty_slots = 0

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area > 2000:  # filter noise
        x, y, w, h = cv2.boundingRect(cnt)

        roi = thresh[y:y+h, x:x+w]
        white_pixels = cv2.countNonZero(roi)

        # If less white pixels → empty slot
        if white_pixels < 1500:
            color = (0, 255, 0)  # Green → Empty
            empty_slots += 1
        else:
            color = (0, 0, 255)  # Red → Occupied

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

# Show result
cv2.putText(image, f"Empty Slots: {empty_slots}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

cv2.imshow("Smart Parking System", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
