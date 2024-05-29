from django.shortcuts import render, redirect
import numpy as np
from sklearn.preprocessing import Normalizer
import cv2
from FaceNet.Models.FaceNet_third_model_proposition_using_preTrained_fineTuning import *
from sklearn.preprocessing import Normalizer
from django.contrib import messages
import pickle
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from .models import *
import datetime
import base64
from pathlib import Path
import pandas as pd
import shutil
# Create your views here.

branchGroupData = {'A':'CSE','B':'CSE','C':'CSE','D':'CSE','E':'ECE','F':'ECE','N':'EE','H':'ME','G':'ME','I':'CE','J':'CE','K':'CHE','L':'BT','M':'PIE'}
branch = ['CSE','ECE','EE','ME','PIE','CE','CHE','BT']
class Summary:
    def __init__(self,name,status):
        self.name = name
        self.status = status

class Attendance:
    def __init__(self,subject="",percentage=0):
        self.subject= subject
        self.percentage = percentage

def home(request):
    date = datetime.date.today()
    year = date.year
    month = date.month
    if(month>=5):  #assuming session changes in August
        yearListA = os.listdir("Attendance Record")
        yearListE = os.listdir('Embeds')
        for i in yearListA:
            if(i<str(year-1)):
                shutil.rmtree(os.path.join("Attendance Record",i)) #We will maintain record till batch an year before the passouts
        
        for i in range(4):
            if str(year+i) not in yearListA:
                os.mkdir(os.path.join("Attendance Record",str(year+i)))
                for j in range(8):
                    branchNow = branch[j]
                    os.mkdir(os.path.join("Attendance Record",str(year+i),branchNow))
                    groups = []
                    for k in list(branchGroupData.keys()):
                        if branchGroupData[k]==branchNow:
                            groups.append(k)
                    
                    for k in groups:
                        os.mkdir(os.path.join("Attendance Record",str(year+i),branchNow,k))
            
            if str(year+i) not in yearListE:
                os.mkdir(os.path.join("Embeds",str(year+i)))

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
                    professor = User.objects._create_user(username,email,pass1)
                    professor.is_staff = True
                    professor.first_name = fname
                    professor.last_name = lname
                    professor.save()
                    prof = Professor(user=professor,username=username, role="professor");
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
                subj = subject.upper()+year+section.upper()
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
                    subjecte = l[pt0]+l[pt1]+l[pt2]
                    subjList.append(subjecte)
                    pt0+=1
                    pt1+=1
                    pt2+=1

                print(subj,subject)
                if subj not in subjList:
                    messages.error(request,"Sorry! You Don't Have This Subject Registered!")
                    return redirect('/')

                path_of_embed = "Embeds\\"+year+"\\"+section+".pkl"

                
                with open(path_of_embed, "rb") as f:
                    data = pickle.load(f)


                class_name_and_arrays = data
                l2_normalizer = Normalizer('l2')
                detector = cv2.CascadeClassifier('FaceNet\\haarcascade_frontalface_default.xml')
                model = Embed_model()
                cap = cv2.VideoCapture(0)

                #Marking Attendance
                df = pd.read_csv(os.path.join("Attendance Record",year,branchGroupData[section],section,subject+".csv"))
                cols = df.columns
                date = datetime.date.today()


                if str(date) not in cols:
                    print(True)
                    df.insert(len(cols),str(date),np.NaN)
                
                rolls = []
                for i in df.Roll:
                    rolls.append(i)

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
                                
                            if int(predicted_class_name) in rolls:
                                row = df[df['Roll']==int(predicted_class_name)].index[0]
                                col = str(date)
                                df.loc[row,col] = 'p'

                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, predicted_class_name.upper(), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)



                    except:
                        pass

                    cv2.imshow('Webcam Feed', frame)


                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        df.fillna('a',inplace=True)
                        df.to_csv(os.path.join("Attendance Record",year,branchGroupData[section],section,subject+".csv"),index=False)
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
                # masterPath = os.path.join("Attendance Record",year,branchGroupData[section],section)
                # masterPathe = Path(masterPath)
                subCheck = subject

                # if(masterPathe.exists()!=True):
                #     messages.error(request,"Given Configuration Does Not Exist!")
                #     return redirect("/")
                

                Prof = Professor.objects.filter(user=request.user)
                Prof=Prof[0]
                subjecte = Prof.subjects

                if subjecte!='':
                    l = subjecte.split(',')
                    pt1 = 0
                    pt2 = 0
                    for i in range(len(l)):
                        print(l[i][0]=="2" and pt1==0)
                        if(l[i][0]=="2" and pt1==0):
                            pt1=i
                        
                        if(len(l[i])==1):
                            print(i,l[i])
                            pt2=i
                            break
                    l.insert(pt1,subject)
                    l.insert(pt2+1,year)
                    l.insert(len(l),section)
                    subj = ','.join(l)
                    Prof.subjects = subj
               
                else:
                    subj = subject.upper()+","+year+','+section.upper()
                    Prof.subjects = subj
        
                Prof.save()

                subjec = subCheck.split(',')
                years = year.split(',')
                sections = section.split(',')

                for i in range(len(subjec)):
                    path = os.path.join("Attendance Record\\",years[i],branchGroupData[sections[i]],sections[i],subjec[i]+".csv")
                    pathe = Path(path)

                    if(pathe.exists()==False):
                        usernames = []
                        objList = Student.objects.filter(year=years[i]).filter(section=sections[i])
                        for i in objList:
                            usernames.append(i.username)
                        
                        df = pd.DataFrame({"Roll":usernames})
                        df.to_csv(path,index=False)
                    

                messages.success(request,"Subjects Registered!")
                return render(request,"attendancePortal.html")
        return render(request,'register.html')
    
    else:
        messages.error(request, "Please LogIn First!!")
        return redirect("/profLogin/")


def makeEmbed(request):
    if request.method=="POST":
        regNo = request.POST.get('regNo')
        year = request.POST.get('year')
        section = request.POST.get('section')
        count = request.POST.get('count')
        image = request.POST.get('image')

        
        image = image[21:]
        tempLocation = "temp"
        path = tempLocation+"\\"+regNo+".png"


        if regNo in os.listdir(tempLocation):
            messages.error(request,"User already in embed")
            return redirect("/")

        image = base64.b64decode(image)


        with open(path,"wb") as f:
            f.write(image)


        count = int(count)
        count-=1
        if count==0:
            try:
                os.mkdir(os.path.join("Embeds",year))
            except:
                pass

            path_of_pkl = os.path.join("Embeds",year,section+".pkl")
            data = []
            l2_normalizer = Normalizer('l2')
            detector = cv2.CascadeClassifier('FaceNet\\haarcascade_frontalface_default.xml')
            model = Embed_model()

            failed = []

            filePath = Path(path_of_pkl)
            if(filePath.exists()):
                with open(path_of_pkl,"rb") as f:
                    class_name_and_array = pickle.load(f)

                for i in class_name_and_array:
                    data.append(i)
                    
            imagePaths = os.listdir(tempLocation)
            for i in imagePaths:
                image = cv2.imread(tempLocation+"//"+i)
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                except:
                    gray = image
                try:
                    (x, y, w, h) = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)[0]
                    face_img = image[y:y+h, x:x+w]
                except:
                    failed.append(i[0:len(i)-4])
                    continue
                face_img = cv2.resize(face_img, (160, 160))
                face_img = face_img.astype('float32') / 255.0
                face_img = np.expand_dims(face_img, axis=0)

                # Get the face embedding
                face_embedding = model.predict(face_img)[0]
                face_embedding = l2_normalizer.transform(np.expand_dims(face_embedding, axis=0))[0]
                data.append((face_embedding,i[0:len(i)-4]))

            with open(os.path.join("Embeds",year,section+".pkl"),'wb') as f:
                pickle.dump(data,f)

            for i in imagePaths:
                path = os.path.join(tempLocation,i)
                os.remove(path)

            if len(failed)==0:
                messages.success(request,"All Students Were Recorded Successfully!")
            
            else:
                strlist = ""
                for i in failed:
                    strlist+=i
                messages.success(request, "Following students were not registered as we couldn't detect faces: "+strlist)
            return redirect('/')
        return render(request,"makeEmbeds.html",{"year":year,"section":section,"count":count})

    return render(request,"makeEmbeds.html")

def counts(request):
    if request.method=="POST":
        year = request.POST.get('year')
        section = request.POST.get('section')
        count = request.POST.get('count')
        if int(count)==0:
            messages.error(request,"You entered 0 count!")
            return redirect('/')
        
        yearCheck = datetime.date.today().year

        if int(year)>yearCheck+4:
            messages.error(request,"You've entered an year value beyond the current academic scope")
            return render(request, "addStudentCount.html")

        return render(request,"makeEmbeds.html",{"year":year,"section":section,"count":count})

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
        if(len(obj)==0):
            messages.error(request,"You are not authorized to access this page")
            return redirect("/")
        
        obj = obj[0]
        attendance_path = os.path.join("Attendance Record",obj.year,branchGroupData[obj.section],obj.section)
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

def Logout(request):
    x = logout(request)
    messages.success(request,"Successfully Logged Out!")
    return redirect("/")