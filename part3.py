import cv2 
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'final_project/faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
Training_Data, Labels = [], []

#파일 개수 만큼 루프 돌리기 
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if images is None:
        continue
    Training_Data.append(np.asarray(images, dtype = np.uint8))
    Labels.append(i)

#훈련할 데이터가 없다면 종료.
if len(Labels)==0:
    print("There is no data to train")
    exit()

Labels = np.asarray(Labels, dtype=np.int32)

#모델 생성 
model = cv2.face.LBPHFaceRecognizer_create()

#학습 시작 
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete!!!")

#########################################

face_classifier = cv2.CascadeClassifier('final_project/haarcascade_frontalface_default.xml')
#전체 사진에서 얼굴 부위만 잘라 리턴 
def face_extractor(img, size=0.5):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img, []
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 255), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))    
    return img, roi
    #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달

##########################################


frame=cv2.imread('final_project/yjg3.jpg', cv2.IMREAD_COLOR) 

image, face = face_extractor(frame)
try:
    face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
    result = model.predict(face)
    #result[1]은 신뢰도이고 0에 가까울 수록 자신과 같다는 뜻이다.
    if result[1] < 500:
        confidence = int(100*(1-(result[1])/300))
        display_string = str(confidence)+'% confidence it it user'
    cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX,
                                                    1,(250, 120, 255),2)

    #80보다 크면 동일 인물로 간주하기 
    if confidence>=80:
        cv2.putText(image, "Matched", (250, 450), cv2.FONT_HERSHEY_COMPLEX,
                                                     1, (0, 255, 0), 2 )
        cv2.imshow('Face Cropper', image)
    else:
        cv2.putText(image, "Un Matched", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 
                                                    1, (0, 0, 255), 2)
        cv2.imshow('Face Cropper', image)
    print('@')
    

except:
    cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX,
                                                    1, (255, 0, 0), 2)
    cv2.imshow('Face Cropper', image)
    pass

cv2.waitKey(0)