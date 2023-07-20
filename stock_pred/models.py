from django.db import models

# Create your models here.
from django.contrib.auth.models import User

class Stock(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.symbol


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    stocks = models.ManyToManyField(Stock, blank=True)

    def __str__(self):
        return self.user.username
    


