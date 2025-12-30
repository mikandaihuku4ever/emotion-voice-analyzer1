import sys
import wave
import numpy as np
import Vokaturi
import os
import ctypes
import speech_recognition as sr
from transformers import pipeline

# Vokaturiライブラリの読み込み
dll_path = os.path.join(os.path.dirname(__file__), "OpenVokaturi-4-0-win64.dll")
Vokaturi.load(dll_path)

# WAVファイルの読み込み
file_name = "output.wav"
print(f"=== 音声・テキスト統合分析 ===")
print(f"分析対象: {file_name}\n")

# ========================================
# 1. 音声から感情を抽出（Vokaturi）
# ========================================
print("【ステップ1】音声感情分析中...")

with wave.open(file_name, 'r') as wav_file:
    num_channels = wav_file.getnchannels()
    sample_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    buffer = wav_file.readframes(num_frames)
    
    if num_channels == 1:
        audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float64)
    else:
        audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float64)
        audio_data = audio_data.reshape(-1, num_channels).mean(axis=1)

# Vokaturiで音声感情分析
voice = Vokaturi.Voice(sample_rate, len(audio_data), True)
buffer_pointer = audio_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
voice.fill_float64array(len(audio_data), buffer_pointer)

quality = Vokaturi.Quality()
emotion_probabilities = Vokaturi.EmotionProbabilities()
voice.extract(quality, emotion_probabilities)

# 音声感情の結果
voice_emotions = {
    "中立": emotion_probabilities.neutrality,
    "幸福": emotion_probabilities.happiness,
    "悲しみ": emotion_probabilities.sadness,
    "怒り": emotion_probabilities.anger,
    "恐怖": emotion_probabilities.fear
}

dominant_voice_emotion = max(voice_emotions, key=voice_emotions.get)
print(f"✓ 完了: 音声感情は「{dominant_voice_emotion}」({voice_emotions[dominant_voice_emotion]:.3f})\n")

# ========================================
# 2. 音声認識でテキスト化
# ========================================
print("【ステップ2】音声認識中...")

recognizer = sr.Recognizer()
with sr.AudioFile(file_name) as source:
    audio = recognizer.record(source)
    
try:
    # Google Speech Recognition（日本語）
    text = recognizer.recognize_google(audio, language="ja-JP")
    print(f"✓ 完了: 認識されたテキスト")
    print(f"  「{text}」\n")
except sr.UnknownValueError:
    print("✗ エラー: 音声を認識できませんでした")
    text = ""
except sr.RequestError as e:
    print(f"✗ エラー: Google Speech Recognition APIに接続できません: {e}")
    text = ""

# ========================================
# 3. テキストから感情を分析
# ========================================
if text:
    print("【ステップ3】テキスト感情分析中...")
    
    # 日本語感情分析モデル（cl-tohoku/bert-base-japanese-sentimentなど）
    # または英語の場合はdistilbert-base-uncased-finetuned-sst-2-english
    try:
        # 簡易的に英語モデルを使用（日本語の場合は日本語対応モデルに変更可能）
        sentiment_analyzer = pipeline("sentiment-analysis", 
                                    model="distilbert-base-uncased-finetuned-sst-2-english")
        
        sentiment_result = sentiment_analyzer(text)[0]
        text_sentiment = sentiment_result['label']  # POSITIVE or NEGATIVE
        text_confidence = sentiment_result['score']
        
        print(f"✓ 完了: テキスト感情は「{text_sentiment}」(信頼度: {text_confidence:.3f})\n")
    except Exception as e:
        print(f"✗ テキスト感情分析でエラー: {e}")
        text_sentiment = "不明"
        text_confidence = 0.0
else:
    print("【ステップ3】スキップ: テキストが認識されませんでした\n")
    text_sentiment = "不明"
    text_confidence = 0.0

# ========================================
# 4. 統合分析・推論
# ========================================
print("=" * 50)
print("【統合分析結果】")
print("=" * 50)

print(f"\n📊 音声感情:")
for emotion, value in voice_emotions.items():
    bar = "█" * int(value * 20)
    print(f"  {emotion:6s}: {bar} {value:.3f}")

print(f"\n🎤 認識テキスト: 「{text}」")
print(f"📝 テキスト感情: {text_sentiment} (信頼度: {text_confidence:.3f})")

# 推論ロジック
print(f"\n🔍 推論:")

if text and text_sentiment != "不明":
    # 音声感情とテキスト感情の不一致を検出
    voice_is_negative = dominant_voice_emotion in ["怒り", "悲しみ", "恐怖"]
    text_is_positive = text_sentiment == "POSITIVE"
    
    if voice_is_negative and text_is_positive:
        print(f"  ⚠️  音声は「{dominant_voice_emotion}」だが、言葉は肯定的")
        print(f"  💡 推理: 感情を抑えている、または本音と建前が異なる可能性")
    elif not voice_is_negative and not text_is_positive:
        print(f"  ⚠️  音声は穏やかだが、言葉は否定的")
        print(f"  💡 推理: 冷静に批判している、または諦めの状態")
    else:
        print(f"  ✅ 音声感情とテキスト感情が一致しています")
        print(f"  💡 推理: 素直な感情表現、言動が一致している")
    
    # 具体的なパターン分析
    if dominant_voice_emotion == "怒り" and voice_emotions["怒り"] > 0.7:
        if text_is_positive:
            print(f"  🔥 強い怒りを感じながらも丁寧な言葉遣い → 必死に抑制している")
        else:
            print(f"  🔥 怒りの感情と否定的な言葉が一致 → 率直な怒りの表現")
    
    elif dominant_voice_emotion == "悲しみ" and voice_emotions["悲しみ"] > 0.7:
        if text_is_positive:
            print(f"  😢 悲しみを隠して明るく振る舞おうとしている")
        else:
            print(f"  😢 悲しみが言葉にも表れている")
    
    elif dominant_voice_emotion == "幸福" and voice_emotions["幸福"] > 0.7:
        if not text_is_positive:
            print(f"  😊 明るい声で皮肉や批判を言っている → 皮肉な表現の可能性")
        else:
            print(f"  😊 純粋に喜んでいる状態")

else:
    print(f"  ⚠️  テキスト分析ができないため、音声感情のみの結果です")
    print(f"  音声からは「{dominant_voice_emotion}」の感情が検出されました")

print("\n" + "=" * 50)

# リソース解放
voice.destroy()
