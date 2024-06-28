from django.urls import path
from . import views
from .views import detect_emotion, save_faceimage, face_recognition

urlpatterns = [
    path('detect_emotion/', detect_emotion, name='detect_emotion'),
    path('save_faceimage/', save_faceimage, name='save_faceimage'),
    path('face_recognition/', face_recognition, name='face_recognition'),
]
