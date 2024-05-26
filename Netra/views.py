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
# Create your views here.

globalSubjectCount = 0

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
                    professor = User.objects.create_user(username,email,pass1)
                    professor.first_name = fname
                    professor.last_name = lname
                    professor.save()
                    prof = Professor(user=professor);
                    prof.save()
                    messages.success(request,"We Have Verified Your Request, Account Creation Successful!")
                    return redirect("/")
                
        
        return render(request, 'profRegistration.html')


def profLogin(request):
    if request.method =="POST":
        username = request.POST.get('username')
        pass1 = request.POST.get('pass1')
        user = authenticate(username=username, password=pass1)

        if user is not None:
            login(request,user)
            context = {'fname':user.first_name}
            messages.success(request,"Logged-In!")
            return render(request,"attendancePortal.html", context)
        else:
            messages.error(request,"Incorrect Credentials!!")
            return redirect("/")

    return render(request, 'profLogin.html')

def studLogin(request):
    return render(request, 'studLogin.html')

def TakeAttendance(request):
    return render(request, 'camera.html')

def BeginCameraFeed(request):
    if request.user.is_authenticated:
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

            path_of_embed = "A:\\FaceNet Extension\\AttendanceSystem\\Embeds\\"+year+"\\"+section+".pkl"

            
            with open(path_of_embed, "rb") as f:
                data = pickle.load(f)


            class_name_and_arrays = data
            l2_normalizer = Normalizer('l2')
            detector = cv2.CascadeClassifier('A:\FaceNet Extension\AttendanceSystem\FaceNet\haarcascade_frontalface_default.xml')
            model = Embed_model()
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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


            
                cv2.imshow('Webcam Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        return redirect("/")
    
    else:
        messages.error(request, "Please LogIn First!!")
        return redirect("/profLogin/")

    
def Subjects(request):
    if request.user.is_authenticated:
        if request.method=="POST":
            subject = request.POST.get('subjectName')
            year = request.POST.get('year')
            section = request.POST.get('section')
            netSubj = subject.capitalize()+','+year+','+section.upper()
            Prof = Professor.objects.filter(user=request.user)
            Prof=Prof[0]
            Prof.subjects = Prof.subjects+netSubj
            Prof.save()
            messages.success(request,"Subjects Registered!")
            return render(request,"attendancePortal.html")
        return render(request,'register.html')
    
    else:
        messages.error(request, "Please LogIn First!!")
        return redirect("/profLogin/")
