import sys
import wave
import numpy as np
import Vokaturi
import os
import ctypes

# Vokaturiライブラリの読み込み
# Windows: OpenVokaturi-4-0-win64.dll
# Mac: OpenVokaturi-4-0-mac.dylib
# Linux: OpenVokaturi-4-0-linux64.so
dll_path = os.path.join(os.path.dirname(__file__), "OpenVokaturi-4-0-win64.dll")
Vokaturi.load(dll_path)

# WAVファイルの読み込み
file_name = "output.wav"
print(f"分析中: {file_name}")

# WAVファイルを開く
with wave.open(file_name, 'r') as wav_file:
    # WAVファイルの情報を取得
    num_channels = wav_file.getnchannels()
    sample_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    
    print(f"チャンネル数: {num_channels}")
    print(f"サンプリングレート: {sample_rate} Hz")
    print(f"フレーム数: {num_frames}")
    
    # 音声データを読み込み
    buffer = wav_file.readframes(num_frames)
    
    # NumPy配列に変換（int16からfloat64へ）
    if num_channels == 1:
        # モノラル
        audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float64)
    else:
        # ステレオの場合は平均を取ってモノラルに変換
        audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float64)
        audio_data = audio_data.reshape(-1, num_channels).mean(axis=1)

# Vokaturiで音声を分析
# multi_threading: True=マルチスレッド使用, False=シングルスレッド
voice = Vokaturi.Voice(sample_rate, len(audio_data), True)

# 音声データをVokaturiに渡す（ctypesポインタに変換）
buffer_pointer = audio_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
voice.fill_float64array(len(audio_data), buffer_pointer)

# 感情を分析
quality = Vokaturi.Quality()
emotion_probabilities = Vokaturi.EmotionProbabilities()
voice.extract(quality, emotion_probabilities)

# 分析結果を表示
print("\n=== 感情分析結果 ===")
print(f"品質: {quality.valid:.3f}")
print(f"\n各感情の確率:")
print(f"中立 (Neutral): {emotion_probabilities.neutrality:.3f}")
print(f"幸福 (Happy):   {emotion_probabilities.happiness:.3f}")
print(f"悲しみ (Sad):   {emotion_probabilities.sadness:.3f}")
print(f"怒り (Angry):   {emotion_probabilities.anger:.3f}")
print(f"恐怖 (Fear):    {emotion_probabilities.fear:.3f}")

# 最も高い感情を特定
emotions = {
    "中立 (Neutral)": emotion_probabilities.neutrality,
    "幸福 (Happy)": emotion_probabilities.happiness,
    "悲しみ (Sad)": emotion_probabilities.sadness,
    "怒り (Angry)": emotion_probabilities.anger,
    "恐怖 (Fear)": emotion_probabilities.fear
}

dominant_emotion = max(emotions, key=emotions.get)
print(f"\n最も強い感情: {dominant_emotion} ({emotions[dominant_emotion]:.3f})")

# Vokaturiリソースを解放
voice.destroy()
