from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home, name="home"),
    path('profLogin/',views.profLogin,name="profLogin"),
    path('profRegister/',views.prof,name="profRegister"),
    path('studLogin/',views.studLogin,name="studLogin"),
    path('camera/',views.TakeAttendance,name='cameraFeed'),
    path('attendance/',views.BeginCameraFeed,name='BegincameraFeed'),
    path('attendance/',views.BeginCameraFeed,name='BegincameraFeed'),
    path('registerSubjects/',views.Subjects, name="Subject"),
    path('addStudent/',views.makeEmbed, name="addStudent"),
]
