import os
import cv2
import numpy as np
from PIL import Image
import pickle


#finding the path
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(BASE_DIR, "images")

#getting the haarcascade file to detect the face
face_cascade = cv2.CascadeClassifier('C:\\Users\\user\\Downloads\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id =0
label_ids= {}
y_labels = []
x_train=[]

#finding the images in directory
for root, dirs, files in os.walk(image_dir):
      for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                  path=os.path.join(root,file)
                  label = os.path.basename(root).replace(" ", "-").lower()
                  #print(label, path)
                  #giving ids to faces
                  if not label in label_ids:
                        label_ids[label] =current_id
                        current_id+=1
                  id_ =label_ids[label]
                  print(label_ids)
                  pil_image = Image.open(path).convert("L")
                  #resizing the image
                  size =(500,500)
                  final_image =pil_image.resize(size, Image.ANTIALIAS)
                  image_array = np.array(final_image, "uint8")
                  #print(image_array)

                  #finding for facing in the given images
                  faces = face_cascade.detectMultiScale(image_array,1.5,5)
                  #finding the position fo face
                  for (x,y,w,h) in faces:
                        roi =image_array[y:y+h, x:x+w]
                        x_train.append(roi)
                        y_labels.append(id_)

with open("labels.pickle", 'wb') as f:
      pickle.dump(label_ids, f)
#train the machine
recognizer.train(x_train,np.array(y_labels))

#saving the traing file
recognizer.save("trainner.yml")
