from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Professor(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        primary_key=True,
    )
    subjects = models.CharField(default="",max_length=1000)
    role = models.CharField(default="",max_length=1000)
    username = models.CharField(default="",max_length=50)


    def __str__(self):
        return self.user.first_name



class Student(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        primary_key=True,
    )
    section = models.CharField(default="",max_length=100)
    username = models.CharField(default="",max_length=50)
    year = models.CharField(default="",max_length=100)
    role = models.CharField(default="",max_length=100)

    def __str__(self):
        return self.user.first_name

