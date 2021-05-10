from django.db import models


# Create your models here.

class Image(models.Model):
    img = models.CharField(max_length=500, null=True, blank=True)
    create_time = models.TimeField(auto_now=True)