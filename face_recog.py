import cv2 
import pandas as pd 
from tflite_runtime.interpreter import Interpreter
import numpy as np 
import glob

PATH = "./images/"
FACE_DETEC =  cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
KERNEL = 7 

interpreter = Interpreter(model_path='./models/vgg19.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

        
class Recognitor:

    def __init__(self,folder) :
        
        self.folder = folder

    @classmethod
    def process_image(cls,path_image):
        image = cv2.imread(path_image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = FACE_DETEC.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        print(faces)
        x,y,w,h = faces[0]
        face_roi = image[y:y+h,x:x+w]
        resized_face = cv2.resize(face_roi,(224,224))
        blured_image = cv2.blur(resized_face,(3,3))
        blured_image = blured_image.astype(np.float32)
        blured_image =np.expand_dims(blured_image,axis=0)
        interpreter.set_tensor(input_details[0]['index'], blured_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data[0]
    def prepare_image(self,image):
        resized_face = cv2.resize(image,(224,224))
        blured_image = cv2.blur(resized_face,(3,3))
        blured_image = blured_image.astype(np.float32)
        blured_image =np.expand_dims(blured_image,axis=0)
        interpreter.set_tensor(input_details[0]['index'], blured_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data[0]
    
    def load_data(self):
        
        paths =  glob.glob(self.folder+"*")
        df = pd.DataFrame({"paths":paths})
        df["names"] = df["paths"].apply(lambda x : x.split("/")[-1].split(".")[0])
        df["map_features"] = df["paths"].apply(Recognitor.process_image)
        return df 
    def rmse(self,map1 , map2):
        return np.mean((map1 - map2)**2)
    def sigmoid(self,value):
        return 1/(1+np.exp(-value))
    def compare(self,frame,map1):
        map2 = self.prepare_image(frame)
        rmse = self.rmse(map1=map1,map2=map2) 
        return rmse
    def get_image(self,df,image):
        
        df["prob"] = df["map_features"].apply(lambda x: self.compare(image,x))
        df = df.sort_values(by='prob')
        df = df.reset_index(drop=True)
        path_image = df["names"][0]
        prob = df["prob"][0]
        del df["prob"]
        return path_image,prob,

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETEC.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces      

cap = cv2.VideoCapture("1107400923-preview.mp4")
re = Recognitor(folder=PATH)
df =re.load_data()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = detect_faces(frame)
    



    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        name,prob = re.get_image(df,face_roi)
        if prob < 30.0 :
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 250), 2)

            cv2.putText(frame, f'suspect: {name}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Aeroport', frame)

    if cv2.waitKey(1) & 0xFF == 27:  
        break
    
cap.release()
cv2.destroyAllWindows()



    



