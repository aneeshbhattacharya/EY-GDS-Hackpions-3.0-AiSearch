from django.db import models
from django.db.models.fields import BooleanField, CharField, json
from jsonfield import JSONField
# Create your models here.

class Image(models.Model):
    name = CharField(max_length=100)
    data = JSONField(default=[])
    tagged = BooleanField(default=False)
    tags = JSONField(default=[])
    pdf_file = CharField(max_length=500,default="No_Location")
    video_file = CharField(max_length=500, default="No Location")
    image_file = CharField(max_length=500,default="No Location")
    


