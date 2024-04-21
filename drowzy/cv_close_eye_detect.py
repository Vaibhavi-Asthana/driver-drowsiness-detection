import cv2,os,pathlib
from playsound import playsound
eye_cascPath = './drowzy/haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
face_cascPath = './drowzy/haarcascade_frontalface_alt.xml'  #face detect model
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)
COLOR=GREEN=(0, 255, 0)
RED=(0,0, 255)
BOX_SIZE=2
cap = cv2.VideoCapture(0)
root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
bg_path = os.path.join(root_dir, "beep-warning-6387.mp3")
print(bg_path)
bg_path=pathlib.PureWindowsPath(bg_path).as_posix()
print(bg_path)
while 1:
    ret, img = cap.read()
    if ret:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        frame_tmp = img
        # print("Found {0} faces!".format(len(faces)))
        if len(faces) > 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), COLOR, BOX_SIZE)
            #frame_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
            
            frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
            eyes = eyeCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )
            if len(eyes) == 0:
                print('Drowsy!!!')
                
                COLOR=RED
                BOX_SIZE=10
                cv2.putText(img,"DROWSYYY!",(x,y-20),cv2.FONT_HERSHEY_TRIPLEX, 0.9, (36,0,255), 2)
            #     '''if CURR_TIME - START_TIME >= "150sec" and mail_sent== False :
            #         send_mail()
            #         mail_sent=True
                playsound(bg_path)#'./beep-warning-6387.mp3')
                print('playing sound using  playsound') 
            else:
                print('Awake!!!')
                COLOR=GREEN
                BOX_SIZE=2
                cv2.putText(img,"Awake",(x,y-20),cv2.FONT_HERSHEY_TRIPLEX, 0.9, (36,255,0), 2)
                #START_TIME=CURR_TIME
                # mial_sent=False
        frame_tmp = cv2.resize(frame_tmp, (1910, 1000), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Face Recognition', frame_tmp)
        waitkey = cv2.waitKey(1)
        if waitkey == ord('q') or waitkey == ord('Q'):
            cv2.destroyAllWindows()
            break