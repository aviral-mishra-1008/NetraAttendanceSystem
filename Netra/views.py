from django.shortcuts import render, HttpResponse, redirect
from .forms import RegistrationForm
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import numpy as np
from sklearn.preprocessing import Normalizer
import cv2
from FaceNet.Models.FaceNet_third_model_proposition_using_preTrained_fineTuning import *
from sklearn.preprocessing import Normalizer
from django.contrib import messages
import pickle
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from .models import *
import datetime
import base64
from django.http import JsonResponse
import json
import pandas as pd
# Create your views here.

class Summary:
    def __init__(self,name,status):
        self.name = name
        self.status = status

class Attendance:
    def __init__(self,subject="",percentage=0):
        self.subject= subject
        self.percentage = percentage

def home(request):
    return render(request,'home.html')

def prof(request):
        if request.method=="POST":
            username = request.POST.get('IDNum')
            email = request.POST.get('email')
            fname = request.POST.get('fname')
            lname = request.POST.get('lname')
            pass1 = request.POST.get('pass1')
            pass2 = request.POST.get('pass2')

            if pass1!=pass2:
                    messages.error(request, "Passwords Didn't Match, Try Again!")
                    return redirect("/")

            with open("Admin Details\\dummy-prof_details.pkl",'rb') as f:
                valid_users = pickle.load(f)
                legal = False

                for i in valid_users:
                    (Id, registered) = i

                    if not registered:
                        print("not registered and Id: ",Id, type(username))
                        if Id==username:
                            legal=True
                            break
                    
                    if registered and Id==username:
                        messages.error(request, "User Already Registered!!")
                        return redirect("/")
                
                if not legal:
                    messages.error(request, "We couldn't verify your details, please contact team!")
                    return redirect("/")
                
                else:
                    professor = User.objects._create_user(username,email,pass1,True,False)
                    professor.first_name = fname
                    professor.last_name = lname
                    professor.save()
                    prof = Professor(user=professor, role="professor");
                    prof.save()
                    messages.success(request,"We Have Verified Your Request, Account Creation Successful!")
                    return redirect("/")
                
        
        return render(request, 'profRegistration.html')


def profLogin(request):
    if request.method =="POST":
        username = request.POST.get('username')
        pass1 = request.POST.get('pass1')
        try:
            obj = Professor.objects.filter(username=username)
            obj[0].role
        except:
            messages.error(request,"You Are Not Registered As A Professor")
            return redirect('/')
        user = authenticate(username=username, password=pass1)

        if user is not None:
            login(request,user)
            context = {'fname':user.first_name,"Prof":False,"Stud":True}
            messages.success(request,"Logged-In!")
            return render(request,"attendancePortal.html", context)
        else:
            messages.error(request,"Incorrect Credentials!!")
            return redirect("/")

    return render(request, 'profLogin.html')

def TakeAttendance(request):
    return render(request, 'camera.html')

def BeginCameraFeed(request):
    if request.user.is_authenticated:
        if request.user.is_staff:
            if request.method == "POST":
                subject = request.POST.get('subject')
                year = request.POST.get('year')
                section = request.POST.get('section')
                Prof = Professor.objects.filter(user=request.user)
                Prof = Prof[0]
                subjList = []
                subj = subject.capitalize()+year+section.upper()
                l = Prof.subjects.split(',')
                
                pt0 = 0
                pt1 = 0
                pt2 = 0

                for i in range(len(l)):
                    if(l[i][0]=='2' and l[i][1]=='0' and pt1==0):
                        pt1 = i
                    
                    if(len(l[i])==1 and pt2==0):
                        pt2=i
                
                while(pt2<len(l)):
                    subject = l[pt0]+l[pt1]+l[pt2]
                    subjList.append(subject)
                    pt0+=1
                    pt1+=1
                    pt2+=1

                
                if subj not in subjList:
                    messages.error(request,"Sorry! You Don't Have This Subject Registered!")
                    return redirect('/')

                path_of_embed = "A:\\FaceNet Extension\\Netra_Attendance_System\\Embeds\\"+year+"\\"+section+".pkl"

                
                with open(path_of_embed, "rb") as f:
                    data = pickle.load(f)


                class_name_and_arrays = data
                l2_normalizer = Normalizer('l2')
                detector = cv2.CascadeClassifier('A:\\FaceNet Extension\\Netra_Attendance_System\\FaceNet\\haarcascade_frontalface_default.xml')
                model = Embed_model()
                cap = cv2.VideoCapture(0)

                #Marking Attendance
                df = pd.read_csv("Attendance Record\\"+year+"\\"+section+".csv")
                cols = df.columns
                date = datetime.date.today()            
                if date not in cols:
                    df.insert(len(cols)-1,str(date),np.NaN)
                    

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    try:
                        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                        for (x, y, w, h) in faces:
                            face_img = frame[y:y+h, x:x+w]
                            face_img = cv2.resize(face_img, (160, 160))
                            face_img = face_img.astype('float32') / 255.0
                            face_img = np.expand_dims(face_img, axis=0)

                            face_embedding = model.predict(face_img)[0]
                            face_embedding = l2_normalizer.transform(np.expand_dims(face_embedding, axis=0))[0]

                            highest_similarity = -float('inf')
                            predicted_class_name = None
                            for array, class_name in class_name_and_arrays:
                                similarity = np.dot(face_embedding, array) / (np.linalg.norm(face_embedding) + np.linalg.norm(array))
                                if similarity > highest_similarity:
                                    highest_similarity = similarity
                                    predicted_class_name = class_name

                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            cv2.putText(frame, predicted_class_name.upper(), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

                            print(predicted_class_name, ",", df['Roll'].str.contains(predicted_class_name).any())
                            if df['Roll'].str.contains(predicted_class_name).any():
                                row = df[df['Roll']==predicted_class_name].index[0]
                                col = str(date)
                                df.loc[row,col] = 'p'
                    except:
                        pass

                    cv2.imshow('Webcam Feed', frame)


                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        df.fillna('a',inplace=True)
                        df.to_csv(section+".csv",index=False)
                        break

                cap.release()
                cv2.destroyAllWindows()

                summary = []
                for i,j in zip(df.Roll,df[str(date)]):
                    obj = Summary(i,j)
                    summary.append(obj)
            
            return render(request,"summary.html",{"display":True,"summary":summary})
        
        else:
            messages.error(request, "Please LogIn First!!")
            return redirect("/profLogin/")

    
def Subjects(request):
    if request.user.is_authenticated: 
        if request.user.is_staff:
            if request.method=="POST":
                subject = request.POST.get('subjectName')
                year = request.POST.get('year')
                section = request.POST.get('section')
                Prof = Professor.objects.filter(user=request.user)
                Prof=Prof[0]
                subject = Prof.subjects

                if subject!='':
                    l = subject.split(',')
                    pt1 = 0
                    pt2 = 0
                    for i in range(len(l)):
                        if(l[i][0]=='2' and pt1==0):
                            pt1=i
                        
                        if(len(l[i])==1):
                            pt2=i
                            break
                    
                    l.insert(pt1-1,subject)
                    l.insert(pt2-1,year)
                    l.insert(len(l)-1,section)
                    subj = l.join(',')
                    Prof.subjects = subj
                
                else:
                    subj = subject+','+year+','+section
                    Prof.subjects = subj
                            
                Prof.save()
                messages.success(request,"Subjects Registered!")
                return render(request,"attendancePortal.html")
        return render(request,'register.html')
    
    else:
        messages.error(request, "Please LogIn First!!")
        return redirect("/profLogin/")


def makeEmbed(request):
    return render(request,"makeEmbeds.html")

def counts(request):
    return render(request, "addStudentCount.html")

def studLogin(request):
     if request.method =="POST":
        username = request.POST.get('username')
        pass1 = request.POST.get('pass1')
        try:
            obj = Student.objects.filter(username=str(username))
            obj[0].role
        except:
             messages.error(request,"Sorry! You Are Not A Verified Student")
             return redirect('/')
        
        user = authenticate(username=username, password=pass1)
        if user is not None:
            login(request,user)
            messages.success(request,"Logged-In!")
            return redirect("/percent/")
        else:
            messages.error(request,"Incorrect Credentials!!")
            return redirect("/")
     return render(request, "studLogin.html")

def studRegistration(request):
    if request.method=="POST":
            username = request.POST.get('IDNum')
            email = request.POST.get('email')
            fname = request.POST.get('fname')
            lname = request.POST.get('lname')
            pass1 = request.POST.get('pass1')
            pass2 = request.POST.get('pass2')
            year = request.POST.get('year')
            section = request.POST.get('section')


            if pass1!=pass2:
                    messages.error(request, "Passwords Didn't Match, Try Again!")
                    return redirect("/")

            with open("Admin Details\\dummy-student_details.pkl",'rb') as f:
                valid_users = pickle.load(f)
                legal = False

                for i in valid_users:
                    (Id, registered) = i

                    if not registered:
                        if Id==username:
                            legal=True
                            break
                    
                    if registered and Id==username:
                        messages.error(request, "User Already Registered!!")
                        return redirect("/")
                
                if not legal:
                    messages.error(request, "We couldn't verify your details, please contact team!")
                    return redirect("/")
                
                else:
                    student = User.objects.create_user(username,email,pass1)
                    student.first_name = fname
                    student.last_name = lname
                    student.save()
                    stud = Student(user=student,role="student",section=section,year=year,username=username);
                    stud.save()
                    messages.success(request,"We Have Verified Your Request, Account Creation Successful!")
                    return redirect("/")
                
    return render(request, "studRegistration.html")

def test(request):
    return render(request,"test.html")

def percent(request):
        obj = Student.objects.filter(user=request.user)
        obj = obj[0]
        attendance_path = os.path.join("Attendance Record",obj.year,obj.section)
        subject_list = os.listdir(attendance_path)
        attendanceMasterRecord = []
        for j in subject_list:
            df = pd.read_csv(os.path.join(attendance_path,j))
            count = 0
            present = 0
            index = 0
            record = df.to_numpy()
            for i in range(len(record)):
                if(record[i][0]==int(obj.username)):
                    index = i
                    break
            
            for i in range(len(record[0])):
                value = record[index][i]
                if value=='p':
                    present+=1
                count+=1

            attendance_record =  Attendance(j[:len(j)-4],round((present/count)*100,2))
            attendanceMasterRecord.append(attendance_record)
        return render(request,"percentage.html",{"Attendance":attendanceMasterRecord})





