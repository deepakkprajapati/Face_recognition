# importing modules
import cv2
import numpy
import pickle

faces_cascade=cv2.CascadeClassifier('C:\\Users\\user\\Downloads\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels =pickle.load(f)
	labels ={v:k for k,v in og_labels.items()}

#fetching the video
cap = cv2.VideoCapture("C:\\Users\\user\\Desktop\\ddddd(2)\\videos\\test.mp4")

#reading the video
while(cap.isOpened()):

	#caturing frame by frame
	ret, frame = cap.read()

	#converting the frame in gray scale image
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#fsearching for face in frame
	faces = faces_cascade.detectMultiScale(gray, 1.4, 5)

	#getting the coordinates of the image
	for (x,y,w,h) in faces:
		#print(x,y,w,h)

		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]

		#predicting for found faces
		id_, conf = recognizer.predict(roi_gray)
		if conf>=95 and conf<=100:
			print(id_)
			print(labels[id_])
			#displaying the name on faces
			font =cv2.FONT_HERSHEY_SIMPLEX
			name =labels[id_]
			color =(255,255,255)
			stroke =2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		#creating the rectangle over the faces
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)

	cv2.imshow("frame",frame)

	#press 'q' to exit
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

#release the  cap object
cap.release()

#destroy all the windows
cv2.destroyAllWindows()
