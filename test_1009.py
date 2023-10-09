import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh  # 追加

def main():
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, image = cap.read()
            
            if not ret:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            annotated_image = image.copy()

            # 指定したランドマークのインデックスのリストを作成
            specified_landmarks_indices = [0, 11, 12, 13, 14, 15, 16, 17, 37, 39, 40, 267, 269, 270, 84, 314, 181, 405]

            if results.face_landmarks:
                for i in specified_landmarks_indices:  # 指定したランドマークだけをループ
                    landmark = results.face_landmarks.landmark[i]
                    cv2.circle(annotated_image, (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])), 2, (0, 255, 0), -1)


            
            # Draw hand landmarks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Check if hand is near mouth
            if results.face_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
                mouth_center = [(landmark.x, landmark.y) for landmark in results.face_landmarks.landmark[48:68]] # Extracting mouth landmarks
                hands = []
                if results.left_hand_landmarks:
                    hands.extend(results.left_hand_landmarks.landmark)
                if results.right_hand_landmarks:
                    hands.extend(results.right_hand_landmarks.landmark)

                for hand_landmark in hands:
                    for (mx, my) in mouth_center:
                        if abs(hand_landmark.x - mx) < 0.05 and abs(hand_landmark.y - my) < 0.05:  # Threshold to check closeness
                            cv2.putText(annotated_image, 'Medication detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            break
            
            cv2.imshow('Webcam Feed', annotated_image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
