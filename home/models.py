from django.db import models

# Create your models here.
class Review(models.Model):
    review = models.CharField(max_length=200)
    rating = models.CharField(max_length=5)