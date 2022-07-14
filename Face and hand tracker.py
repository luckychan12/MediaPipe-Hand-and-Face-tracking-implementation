import cv2
import mediapipe as mp

cv2.namedWindow("display")
vc = cv2.VideoCapture(0)
ptime=0
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

mediapipe_draw = mp.solutions.drawing_utils
mediapipe_facemesh = mp.solutions.face_mesh
mediapipe_hands = mp.solutions.hands
mediapipe_objectron = mp.solutions.objectron
mediapipe_drawing_styles = mp.solutions.drawing_styles


hands = mediapipe_hands.Hands(static_image_mode=False, 
                              max_num_hands=2, 
                              model_complexity=1, 
                              min_detection_confidence=0.5, 
                              min_tracking_confidence=0.5)
facemesh = mediapipe_facemesh.FaceMesh(
    max_num_faces=5)
drawing_spec = mediapipe_draw.DrawingSpec(color=(0, 255, 0), 
                                          thickness=1, 
                                          circle_radius=1)

objectron = mediapipe_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.99,
                            model_name='Cup')


while rval:
    cv2.imshow("preview", cv2.flip(frame, 1))
    rval, frame = vc.read()
    frame2 = frame.copy()
    frame.flags.writeable = False
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = facemesh.process(frame_RGB)
    results_hand = hands.process(frame_RGB)
    results_obj = objectron.process(frame_RGB)
    
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mediapipe_draw.draw_landmarks(
                frame2, landmark_list=face_landmarks, 
                connections=mediapipe_facemesh.FACEMESH_CONTOURS, 
                landmark_drawing_spec=drawing_spec)
            
            mediapipe_draw.draw_landmarks(
                image=frame2,
                landmark_list=face_landmarks,
                connections=mediapipe_facemesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=mediapipe_drawing_styles
                .get_default_face_mesh_tesselation_style())
 
        
            
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            mediapipe_draw.draw_landmarks(
                image=frame2, 
                landmark_list=hand_landmarks, 
                connections=mediapipe_hands.HAND_CONNECTIONS, 
                landmark_drawing_spec=mediapipe_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mediapipe_drawing_styles.get_default_hand_connections_style())
    if results_obj.detected_objects:
        for obj_landmarks in results_obj.detected_objects:
            mediapipe_draw.draw_landmarks(
                image=frame2,
                landmark_list=obj_landmarks.landmarks_2d,
                connections=mediapipe_objectron.BOX_CONNECTIONS)
            mediapipe_draw.draw_axis(frame_out, obj_landmarks.rotation, obj_landmarks.translation)
    
    cv2.imshow("display", cv2.flip(frame2, 1))
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("display")
