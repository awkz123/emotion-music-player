# streamlit_emotion_dj.py (using streamlit-webrtc for live video)
import os
import cv2
import pygame
import threading
import random
import csv
from datetime import datetime
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Configuration
title = "🎵 Emotion-Aware Music Player"
BASE_DIR = os.path.dirname(__file__)
SONGS_DIR = os.path.join(BASE_DIR, "songs")
LOG_FILE = os.path.join(BASE_DIR, "emotion_log.csv")

# Load songs per emotion
SONG_MAP = {}
for emo in ["happy", "sad", "angry", "neutral"]:
    folder = os.path.join(SONGS_DIR, emo)
    if os.path.isdir(folder):
        SONG_MAP[emo] = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp3')]

# Ensure at least one song per emotion
for emo, tracks in SONG_MAP.items():
    if not tracks:
        raise FileNotFoundError(f"No mp3 files found in {emo} folder")

# Initialize audio
def init_audio():
    pygame.mixer.init()

class AudioPlayer:
    def __init__(self):
        self.current = None
        self.lock = threading.Lock()
    def play(self, emo):
        with self.lock:
            if emo == self.current:
                return
            pygame.mixer.music.stop()
            track = random.choice(SONG_MAP[emo])
            pygame.mixer.music.load(track)
            pygame.mixer.music.play(-1)
            self.current = emo
    def stop(self):
        pygame.mixer.music.stop()

# Log emotion
def log_emotion(emo):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(['timestamp', 'emotion'])
        writer.writerow([datetime.now().isoformat(), emo])

# Video transformer for webrtc
class EmotionTransformer(VideoTransformerBase):
    def __init__(self, player, override):
        self.player = player
        self.override = override
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            res = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            emo = res[0]['dominant_emotion']
        except:
            emo = 'neutral'
        if self.override:
            emo = self.override
        if emo not in SONG_MAP:
            emo = 'neutral'
        self.player.play(emo)
        log_emotion(emo)
        cv2.putText(img, emo.capitalize(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return img

# Streamlit UI
st.title(title)
init_audio()
player = AudioPlayer()
over = st.selectbox("Override Emotion", [None, 'happy','sad','angry','neutral'])

webrtc_streamer(
    key="emotion_dj",
    video_transformer_factory=lambda: EmotionTransformer(player, over),
    media_stream_constraints={"video": True, "audio": False}
)