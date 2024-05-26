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

    def __str__(self):
        return self.user.first_name

