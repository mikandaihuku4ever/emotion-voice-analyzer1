# より精度の高い音声感情分析の選択肢

"""
Vokaturiより精度が高い可能性のある代替手段：

1. **Hugging Face Transformers - wav2vec2ベースのモデル**
   - モデル: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
   - 利点: 最新のディープラーニング、7つの感情を認識
   - 精度: Vokaturiより高精度

2. **librosa + TensorFlow/PyTorch モデル**
   - 音響特徴量（MFCC、スペクトログラム）を抽出
   - カスタムモデルで学習可能

3. **Microsoft Azure Speech Service**
   - 商用だが高精度
   - 多言語対応

4. **日本語特化モデル**
   - 日本語音声に特化した感情認識モデル
   - 例: JTES (Japanese Twitter Emotion Dataset) ベースのモデル
"""

import sys
import wave
import numpy as np
import os
import ctypes
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

print("=" * 60)
print("【高精度版】音声感情分析プログラム")
print("=" * 60)

# WAVファイルの読み込み
file_name = "output.wav"
print(f"\n分析対象: {file_name}")

# WAVファイルを開く
with wave.open(file_name, 'r') as wav_file:
    num_channels = wav_file.getnchannels()
    sample_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    
    print(f"\n📊 音声情報:")
    print(f"  - チャンネル数: {num_channels}")
    print(f"  - サンプリングレート: {sample_rate} Hz")
    print(f"  - 長さ: {num_frames / sample_rate:.2f} 秒")
    
    # 音声データを読み込み
    buffer = wav_file.readframes(num_frames)
    
    # NumPy配列に変換
    if num_channels == 1:
        audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        audio_data = audio_data.reshape(-1, num_channels).mean(axis=1)

print("\n🔄 感情分析モデルを読み込み中...")

# Wav2Vec2ベースの感情認識モデル
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

try:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
    
    # デバイス設定（GPU利用可能ならGPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"✓ モデル読み込み完了 (デバイス: {device})")
    
    # サンプリングレート変換（モデルが16kHzを期待している場合）
    if sample_rate != 16000:
        print(f"\n🔄 サンプリングレート変換中: {sample_rate} Hz → 16000 Hz")
        from scipy import signal
        audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
        sample_rate = 16000
    
    print("\n🔍 感情分析中...")
    
    # 特徴抽出
    inputs = feature_extractor(
        audio_data, 
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )
    
    # 推論
    with torch.no_grad():
        inputs = {key: val.to(device) for key, val in inputs.items()}
        logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # モデルの元の感情ラベル
    model_emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
    
    # モデルの出力を取得
    raw_results = {}
    for idx, emotion in enumerate(model_emotions):
        prob = probabilities[0][idx].item()
        raw_results[emotion] = prob
    
    # カスタム感情マッピング
    # 嫌悪を怒りと恐怖に分散（60%怒り、40%恐怖）
    disgust_prob = raw_results['disgust']
    
    # 喜びを計算（落ち着きと幸福の中間として、両方の影響を受ける）
    # 喜び = (落ち着き * 0.3 + 幸福 * 0.7) の重み付き平均
    joyful_prob = raw_results['calm'] * 0.3 + raw_results['happy'] * 0.7
    
    # 最終的な感情の確率を計算
    results = {
        'angry': raw_results['angry'] + disgust_prob * 0.6,  # 嫌悪の60%を怒りに
        'calm': raw_results['calm'] * 0.7,  # 落ち着きを調整（喜びに一部使用）
        'joyful': joyful_prob,  # 新しい感情「喜び」
        'fearful': raw_results['fearful'] + disgust_prob * 0.4,  # 嫌悪の40%を恐怖に
        'happy': raw_results['happy'] * 0.3,  # 幸福を調整（喜びに一部使用）
        'neutral': raw_results['neutral'],
        'sad': raw_results['sad']
    }
    
    # 日本語ラベル
    emotions_jp = {
        'angry': '怒り',
        'calm': '落ち着き',
        'joyful': '喜び',
        'fearful': '恐怖',
        'happy': '幸福',
        'neutral': '中立',
        'sad': '悲しみ'
    }
    
    # 結果表示
    print("\n" + "=" * 60)
    print("【感情分析結果】")
    print("=" * 60)
    
    # 感情の表示順序を固定（幸福、喜び、落ち着き、中立、悲しみ、怒り、恐怖）
    emotion_order = ['happy', 'joyful', 'calm', 'neutral', 'sad', 'angry', 'fearful']
    
    for emotion in emotion_order:
        prob = results[emotion]
        emotion_jp = emotions_jp[emotion]
        bar = "█" * int(prob * 40)
        print(f"{emotion_jp:8s} ({emotion:8s}): {bar} {prob:.3f}")
    
    # 最も高い感情
    dominant_emotion = max(results, key=results.get)
    dominant_emotion_jp = emotions_jp[dominant_emotion]
    print(f"\n🎯 最も強い感情: {dominant_emotion_jp} ({dominant_emotion})")
    print(f"   信頼度: {results[dominant_emotion]:.1%}")
    
    # 信頼度の評価
    if results[dominant_emotion] > 0.7:
        print("   評価: 高い信頼度 ✅")
    elif results[dominant_emotion] > 0.4:
        print("   評価: 中程度の信頼度 ⚠️")
    else:
        print("   評価: 低い信頼度（複数の感情が混在）⚠️")
    
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ エラーが発生しました: {e}")
    print("\n代替案: Vokaturiを使用します...\n")
    
    # Vokaturiにフォールバック
    import Vokaturi
    
    dll_path = os.path.join(os.path.dirname(__file__), "OpenVokaturi-4-0-win64.dll")
    Vokaturi.load(dll_path)
    
    # Vokaturiで分析
    voice = Vokaturi.Voice(sample_rate, len(audio_data), True)
    buffer_pointer = audio_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    voice.fill_float64array(len(audio_data), buffer_pointer)
    
    quality = Vokaturi.Quality()
    emotion_probabilities = Vokaturi.EmotionProbabilities()
    voice.extract(quality, emotion_probabilities)
    
    print("【Vokaturi分析結果】")
    print(f"中立: {emotion_probabilities.neutrality:.3f}")
    print(f"幸福: {emotion_probabilities.happiness:.3f}")
    print(f"悲しみ: {emotion_probabilities.sadness:.3f}")
    print(f"怒り: {emotion_probabilities.anger:.3f}")
    print(f"恐怖: {emotion_probabilities.fear:.3f}")
    
    voice.destroy()
