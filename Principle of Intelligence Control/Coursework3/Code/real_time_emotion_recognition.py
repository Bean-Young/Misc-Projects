import argparse
import cv2
import torch
import numpy as np
import time
import os
import pygame
import random
from torch import nn
import torchvision.transforms as transforms
from face_detector.face_detector import DnnDetector, HaarCascadeDetector
from model.model import Mini_Xception
from utils import get_label_emotion, histogram_equalization
from face_alignment.face_alignment import FaceAlignment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pygame.mixer.init()


def play_music(emotion, current_emotion):
    if emotion != current_emotion:
        if pygame.mixer.music.get_busy() == 0:
            music_dir = 'music_files'
            music_files = {
                'Happy': ['陶喆 - 小镇姑娘.mp3', 'Steady Me-Hollyn.Aaron Cole#gMLr0.mp3'],
                'Sad': ['郑润泽-于是.mp3', '郭顶 - 水星记.mp3'],
                'Neutral': ['One Time-Marian Hill#1MnTM.mp3', '陶喆 - 爱我还是他.mp3'],
            }
            if emotion in music_files:
                selected_music = random.choice(music_files[emotion])
                music_path = os.path.join(music_dir, selected_music)
                print(f"Playing {emotion} music: {selected_music}")
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.play(0)
            else:
                print("Unknown emotion, no music selected.")
            return emotion
    return current_emotion


def main(args, max_duration=10):
    mini_xception = Mini_Xception().to(device)
    mini_xception.eval()

    checkpoint = torch.load(args.pretrained, map_location=device)
    mini_xception.load_state_dict(checkpoint['mini_xception'])
    face_alignment = FaceAlignment()
    root = 'face_detector'
    face_detector = HaarCascadeDetector(root) if args.haar else DnnDetector(root)

    video = cv2.VideoCapture(0)
    isOpened = video.isOpened()
    print('video.isOpened:', isOpened)

    t1 = 0
    current_emotion = None

    cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_time = time.time()

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= max_duration:
            print("时间到了，自动退出表情识别阶段。")
            break

        _, frame = video.read()
        if not isOpened:
            break

        t2 = current_time
        fps = round(1 / (t2 - t1)) if t2 != t1 else 0
        t1 = t2

        frame = cv2.flip(frame, 1)

        faces = face_detector.detect_faces(frame)
        for face in faces:
            (x, y, w, h) = face

            input_face = face_alignment.frontalize_face(face, frame)
            input_face = cv2.resize(input_face, (48, 48))
            input_face = histogram_equalization(input_face)

            input_face = transforms.ToTensor()(input_face).to(device)
            input_face = torch.unsqueeze(input_face, 0)

            with torch.no_grad():
                input_face = input_face.to(device)
                emotion = mini_xception(input_face)

                softmax = torch.nn.Softmax(dim=1)
                emotions_soft = softmax(emotion).cpu().numpy().squeeze()
                emotion_label = torch.argmax(emotion).item()
                emotion_label = get_label_emotion(emotion_label)
                percentage = round(emotions_soft.max() * 100, 2)

                cv2.putText(frame, f"{emotion_label} ({percentage}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                current_emotion = play_music(emotion_label, current_emotion)

        cv2.putText(frame, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video", frame)

        cv2.waitKey(1)

    print("表情识别阶段完成，切换到手势识别阶段。")
    video.release()
    cv2.destroyAllWindows()
