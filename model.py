import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for fire-like colors in HSV
    lower_red = np.array([0, 100, 100])  # Lower bound for red (fire)
    upper_red = np.array([10, 255, 255])  # Upper bound for red (fire)

    # Create a mask for detecting fire colors (red shades)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # You can combine this with other colors like yellow and orange if needed.
    # For example, yellow and orange ranges can also be added as follows:
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    lower_yellow = np.array([25, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine all masks (red, orange, yellow)
    combined_mask = cv2.bitwise_or(mask, mask_orange)
    combined_mask = cv2.bitwise_or(combined_mask, mask_yellow)

    # Count the number of non-zero pixels (which indicates potential fire)
    fire_pixels = np.count_nonzero(combined_mask)

    # If there are enough fire-like pixels, we classify it as "FIRE DETECTED!"
    if fire_pixels > 500:  # You can adjust this threshold based on your setup
        cv2.putText(frame, 'FIRE DETECTED!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'NO FIRE', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Fire Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
