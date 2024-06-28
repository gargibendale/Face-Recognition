import json
from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
import base64
from django.views.decorators.csrf import csrf_exempt
from deepface import DeepFace
import cv2
import numpy as np
import os
from django.core.files.storage import default_storage
from django.conf import settings

import logging

logger = logging.getLogger(__name__)

@csrf_exempt
def detect_emotion(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image', '')

            if not image_data:
                return JsonResponse({'status': 'fail', 'message': 'Image not provided'}, status=400)

            image_data = base64.b64decode(image_data)
            file_bytes = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                return JsonResponse({'status': 'fail', 'message': 'Invalid image data'}, status=400)

            new_image_path = os.path.join(settings.MEDIA_ROOT, 'detect_emotion.jpg')
            cv2.imwrite(new_image_path, img)

            objs = DeepFace.analyze(
                img_path=new_image_path, 
                actions=['emotion']
            )

            emotion_results = objs[0]['dominant_emotion']
            return JsonResponse({'status': 'success', 'message': emotion_results})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@csrf_exempt
def save_faceimage(request):
    if request.method == 'POST':
        if 'image' not in request.FILES or 'name' not in request.POST:
            return JsonResponse({'status': 'fail', 'message': 'Image or name not provided'}, status=400)
        
        file = request.FILES['image']
        user_name = request.POST['name']
        
        if not file or not user_name:
            return JsonResponse({'status': 'fail', 'message': 'Invalid image data or name'}, status=400)
        
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                return JsonResponse({'status': 'fail', 'message': 'Invalid image data'}, status=400)
            
            new_image_path = os.path.join(settings.MEDIA_ROOT, f'{user_name}.jpg')
            cv2.imwrite(new_image_path, img)
            return JsonResponse({'status': 'success', 'message': f'Image saved as {new_image_path}'})
        
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@csrf_exempt
def face_recognition(request):
    if request.method == 'POST':
        if 'image' not in request.FILES or 'name' not in request.POST:
            return JsonResponse({'status': 'fail', 'message': 'Image or name not provided'}, status=400)
        
        file = request.FILES['image']
        user_name = request.POST['name']
        
        if not file or not user_name:
            return JsonResponse({'status': 'fail', 'message': 'Invalid image data or name'}, status=400)
        
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                return JsonResponse({'status': 'fail', 'message': 'Invalid image data'}, status=400)
            
            captured_image_path = os.path.join(settings.MEDIA_ROOT, 'captured_image.jpg')
            cv2.imwrite(captured_image_path, img)

            known_image_path = os.path.join(settings.MEDIA_ROOT, f'{user_name}.jpg')
            
            result = DeepFace.verify(img1_path=captured_image_path, img2_path=known_image_path)
            
            if result['verified']:
                return JsonResponse({'status': 'success', 'message': 'Face recognized'})
            else:
                return JsonResponse({'status': 'fail', 'message': 'Face not recognized'})
        
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

