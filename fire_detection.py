import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('fire_detection_model.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    img = cv2.resize(frame, (224, 224)) 
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  


    prediction = model.predict(img)


    confidence = prediction[0][0] 

    if confidence > 0.9:  
        fire_score = confidence * 100
        cv2.putText(frame, f'FIRE DETECTED! ({fire_score:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        no_fire_score = (1 - confidence) * 100
        cv2.putText(frame, f'NO FIRE ({no_fire_score:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Fire Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
