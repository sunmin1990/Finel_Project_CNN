import cv2 
import matplotlib.pyplot as plt
import numpy  as np 
import os
import sys
#얼굴 인식용 xml파일 


face_classifier = cv2.CascadeClassifier('final_project/haarcascade_frontalface_default.xml')
#전체 사진에서 얼굴 부위만 잘라 리턴 
def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face
# cap = cv2.VideoCapture(0)
count = 0
while True:
    # ret, frame = cap.read()
    frames = os.listdir('final_project/pictures')
    stop = len(frames)
    for f in frames:
        f = 'final_project/pictures/'+ f
        frame = cv2.imread(f,cv2.IMREAD_COLOR)
        if face_extractor(frame) is not None:
            count+=1
            face = cv2.resize(face_extractor(frame),(200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = 'final_project/faces/user'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass
        if cv2.waitKey(1)==13 or count==stop:
            break
# cap.release()
#     cv2.destroyAllWindows()
    print('Colleting Samples Complete!!!')
    sys.exit()



# def face_extractor(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, 1.3, 5)
#     if faces is():
#         return None
#     for(x, y, w, h) in faces:
#         cropped_face = img[y:y+h, x:x+w]
#     return cropped_face
# count = 0

# f1 = cv2.imread('final_project/pictures/img_180.jpg',cv2.IMREAD_COLOR)
# print(face_extractor(f1))
# print('-'*50)

# while True:
#     frames = os.listdir('final_project/pictures')
#     stop = len(frames)   
#     for f in frames:
#         f = 'final_project/pictures/'+ f                
#         frame = cv2.imread(f,cv2.IMREAD_COLOR)
#         print(frame)        
#         #얼굴 감지하여 얼굴만 가져오기     
#         print(face_extractor(frame))
#         sys.exit()
#         if face_extractor(frame) is not None:
#             count +=1
#             #얼굴 이미지 크기를 200x200으로 조정 
#             face = cv2.resize(face_extractor(frame), (200, 200))
#             #resize된 이미지를 흑백으로 변환 
#             face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#             #faces폴더에 jpg파일로 저장 
#             file_name_path = 'final_project/faces/user'+str(count)+'.jpg'  
#             print(file_name_path)
#             cv2.imwrite(file_name_path, face)
#             #화면에 얼굴과 현재 저장 개수 표시 
#             cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#             cv2.imshow('Face Cropper', face)
#         else:
#             print("Face not Found")
#             # sys.exit()
#             pass
#         if cv2.waitKey(1) or count==stop:
#             break
#         print('Collecting Samlples Complete!!!')    
#         sys.exit()