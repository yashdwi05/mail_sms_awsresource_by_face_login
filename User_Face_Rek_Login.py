#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from twilio.rest import Client
from os import listdir, environ, system
from os.path import isfile, join
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from getpass import getpass
from subprocess import getoutput
from time import sleep


# Get the training data we previously made
data_path = './facedata/yash/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
# pip install opencv-contrib-python

yash_model  = cv2.face_LBPHFaceRecognizer.create()
# Let's train our model 
yash_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")


# Get the training data we previously made
data_path = './facedata/himani/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

tanvi_model  = cv2.face_LBPHFaceRecognizer.create()
# Let's train our model 
tanvi_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")


#Function for sending SMS
def sms(user):
    sid = environ['TWILIO_ACCOUNT_SID']
    token = environ['TWILIO_AUTH_TOKEN']
    client = Client(sid, token)

    message = client.messages.create(
                              body='Hey! ' + user + ' Logged in By face.',
                              from_='+18594487214',
                              to='+918824881106'
    )


#Sending Mail
def email(user, sender_pass):
    mail_content = "Hey!! Face Login By " + user
    
    #The mail addresses and password
    sender_address = 'yashdwivedi3150@gmail.com'
    #sender_pass = getpass("Enter Mail Password: ")
    
    print(sender_pass)
    receiver_address = 'dwiyash23@outlook.com'

    #Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'Login Info'
    message.attach(MIMEText(mail_content, 'plain'))

    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string(message)
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')


#Function for Launching AWS environment
def environment():
    instance = getoutput("aws ec2 run-instances --image-id ami-0ad704c126371a549 --count 1  --placement AvailabilityZone=ap-south-1b --instance-type t2.micro --key-name MachineLearning --query Instances[].InstanceId")
    vol_id = getoutput("aws ec2 create-volume --availability-zone ap-south-1b --size 5 --query VolumeId")
    sleep(30)
    getoutput("aws ec2 attach-volume --instance-id {} --volume-id {} --device /dev/sdf".format(instance[2:27], vol_id))
    print("Environment Launched!!")



#Face Recognition

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

#sender_pass = getpass("Enter Sender Mail Password: ")
sender_pass = environ['GMAIL_AUTH_PASS']

# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        result_yash = yash_model.predict(face)
        result_tanvi = tanvi_model.predict(face)
        # harry_model.predict(face)
        
        if result_yash[1] < 500:
            confidence1 = int( 100 * (1 - (result_yash[1])/400) )
            # display_string = str(confidence) + '% Confident it is Yash'
            
        # cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,120,150), 2)
        
        if result_tanvi[1] < 500:
            confidence = int( 100 * (1 - (result_tanvi[1])/400) )
            # display_string = str(confidence) + '% Confident it is Himani'
            
        # cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,120,150), 2)
        
        if confidence1 > 90:
            cv2.putText(image, "Hey Yash", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            email("Yash", sender_pass)
            sms("Yash")
            email = True
            break
            
        elif confidence > 90:
            cv2.putText(image, "Hey Himani", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            email("Tanvi", sender_pass)
            environment()
            email = True
            break
         
        else:
            
            cv2.putText(image, "Not a User", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
        
        if email == True:
            break

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()     


