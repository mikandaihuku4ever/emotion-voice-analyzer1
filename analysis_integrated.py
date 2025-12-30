import sys
import wave
import numpy as np
import os
import ctypes
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import speech_recognition as sr
from transformers import pipeline

print("=" * 70)
print("【統合版】音声+テキスト感情分析プログラム")
print("=" * 70)

# WAVファイルの読み込み
file_name = "output.wav"
print(f"\n📁 分析対象: {file_name}")

# ========================================
# ステップ1: 音声から感情を抽出
# ========================================
print("\n" + "=" * 70)
print("【ステップ1】音声感情分析")
print("=" * 70)

with wave.open(file_name, 'r') as wav_file:
    num_channels = wav_file.getnchannels()
    sample_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    
    print(f"  チャンネル数: {num_channels}")
    print(f"  サンプリングレート: {sample_rate} Hz")
    print(f"  長さ: {num_frames / sample_rate:.2f} 秒")
    
    buffer = wav_file.readframes(num_frames)
    
    if num_channels == 1:
        audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        audio_data = audio_data.reshape(-1, num_channels).mean(axis=1)

print("\n🔄 モデル読み込み中...")

model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# サンプリングレート変換
if sample_rate != 16000:
    from scipy import signal
    audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
    sample_rate = 16000

# 音声感情分析
inputs = feature_extractor(audio_data, sampling_rate=sample_rate, return_tensors="pt", padding=True)

with torch.no_grad():
    inputs = {key: val.to(device) for key, val in inputs.items()}
    logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

# モデルの元の感情ラベル
model_emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad']

raw_results = {}
for idx, emotion in enumerate(model_emotions):
    prob = probabilities[0][idx].item()
    raw_results[emotion] = prob

# カスタム感情マッピング
disgust_prob = raw_results['disgust']
joyful_prob = raw_results['calm'] * 0.3 + raw_results['happy'] * 0.7

# 中立を振り分け（ポジティブ→落ち着き、ネガティブ→興奮）
neutral_prob = raw_results['neutral']
neutral_to_calm = neutral_prob * 0.6  # 中立の60%を落ち着きに
neutral_to_excitement = neutral_prob * 0.4  # 中立の40%を興奮(ネガティブ)に

# 怒りを「本当の怒り」と「興奮(ネガティブ)」に分ける
anger_prob = raw_results['angry'] + disgust_prob * 0.6
excitement_from_anger = anger_prob * 0.5  # 怒りの50%を興奮に
real_anger_prob = anger_prob * 0.5  # 怒りの50%を本当の怒りとして残す

# 興奮をネガティブな感情として統合（イライラ、焦り、不安定な状態）
total_excitement = excitement_from_anger + neutral_to_excitement

voice_emotions = {
    'happy': raw_results['happy'] * 0.3,
    'joyful': joyful_prob,
    'calm': raw_results['calm'] * 0.7 + neutral_to_calm,  # 中立のポジティブ部分を追加
    'excitement': total_excitement,  # ネガティブな興奮（イライラ、焦り）
    'angry': real_anger_prob,
    'angry': real_anger_prob,
    'sad': raw_results['sad'],
    'fearful': raw_results['fearful'] + disgust_prob * 0.4
}

print("✓ 音声感情分析完了")

# ========================================
# ステップ2: 音声認識でテキスト化
# ========================================
print("\n" + "=" * 70)
print("【ステップ2】音声認識")
print("=" * 70)

recognizer = sr.Recognizer()
text = ""

try:
    with sr.AudioFile(file_name) as source:
        audio = recognizer.record(source)
    text = recognizer.recognize_google(audio, language="ja-JP")
    print(f"✓ 認識されたテキスト:")
    print(f"  「{text}」")
except sr.UnknownValueError:
    print("✗ 音声を認識できませんでした")
except sr.RequestError as e:
    print(f"✗ エラー: {e}")

# ========================================
# ステップ3: テキスト感情分析（日本語対応）
# ========================================
print("\n" + "=" * 70)
print("【ステップ3】テキスト感情分析")
print("=" * 70)

text_emotions = {}

if text:
    try:
        # 日本語感情分析（多クラス分類）
        # より詳細な分析のため、感情ごとにキーワード検出も行う
        
        # ポジティブ/ネガティブ判定
        sentiment_analyzer = pipeline("sentiment-analysis", 
                                    model="distilbert-base-uncased-finetuned-sst-2-english")
        
        sentiment_result = sentiment_analyzer(text)[0]
        is_positive = sentiment_result['label'] == 'POSITIVE'
        confidence = sentiment_result['score']
        
        print(f"  基本感情: {sentiment_result['label']} (信頼度: {confidence:.3f})")
        
        # キーワードベースの感情検出（日本語対応）
        keywords = {
            'happy': ['嬉しい', '幸せ', '楽しい', '良い', 'いい', '最高', '素晴らしい', 'ありがとう', 'よかった'],
            'joyful': ['喜び', '喜ぶ', 'わくわく', 'ワクワク', '楽しみ', '面白い', 'うれしい', 'すごい', 'やった', 'わあ'],
            'calm': ['落ち着', '穏やか', '平和', '安心', 'リラックス', '静か', 'ゆっくり', '普通', 'まあまあ'],
            'excitement': ['イライラ', 'ソワソワ', '焦', '落ち着かない', 'バタバタ', '慌て', '急', '忙しい', '追われ'],
            'angry': ['怒', '腹立', 'むかつ', '許せない', 'ふざけるな', 'うるさい'],
            'sad': ['悲しい', '寂しい', '辛い', '残念', '泣', '涙', '苦しい', '悲'],
            'fearful': ['怖い', '不安', '心配', '恐ろしい', 'ドキドキ', '緊張', '震え']
        }
        
        # キーワードマッチング
        keyword_scores = {emotion: 0 for emotion in keywords.keys()}
        for emotion, words in keywords.items():
            for word in words:
                if word in text:
                    keyword_scores[emotion] += 1
        
        # 強い感情のキーワードが検出されたか確認
        negative_keyword_count = keyword_scores['sad'] + keyword_scores['angry'] + keyword_scores['fearful']
        positive_keyword_count = keyword_scores['happy'] + keyword_scores['joyful']
        excitement_keyword_count = keyword_scores['excitement']
        
        print(f"  検出されたキーワード:")
        print(f"    ポジティブ: {positive_keyword_count}個")
        print(f"    興奮: {excitement_keyword_count}個")
        print(f"    ネガティブ: {negative_keyword_count}個")
        
        # ポジティブキーワードがある場合は、ネガティブを完全に排除
        has_positive_keywords = positive_keyword_count > 0
        has_negative_keywords = negative_keyword_count > 0
        has_excitement_keywords = excitement_keyword_count > 0
        
        # テキスト感情スコアの計算（中立を排除）
        if has_positive_keywords and not has_negative_keywords and not has_excitement_keywords:
            # ポジティブキーワードのみ、興奮なし → ポジティブな感情のみ
            text_emotions = {
                'happy': 0.30 + keyword_scores['happy'] * 0.20,
                'joyful': 0.35 + keyword_scores['joyful'] * 0.20,
                'calm': 0.35 + keyword_scores['calm'] * 0.20,
                'excitement': 0.0,
                'angry': 0.0,
                'sad': 0.0,
                'fearful': 0.0
            }
        elif has_positive_keywords and has_excitement_keywords and not has_negative_keywords:
            # ポジティブ+興奮(ネガティブ) → 混在状態
            text_emotions = {
                'happy': 0.20 + keyword_scores['happy'] * 0.15,
                'joyful': 0.25 + keyword_scores['joyful'] * 0.15,
                'calm': 0.15 + keyword_scores['calm'] * 0.10,
                'excitement': 0.25 + keyword_scores['excitement'] * 0.20,
                'angry': 0.08,
                'sad': 0.05,
                'fearful': 0.02
            }
        elif has_negative_keywords and not has_positive_keywords:
            # ネガティブキーワードのみ → ネガティブな感情
            text_emotions = {
                'happy': 0.0,
                'joyful': 0.0,
                'calm': 0.15,
                'excitement': 0.15 + keyword_scores['excitement'] * 0.15,
                'angry': 0.30 + keyword_scores['angry'] * 0.20,
                'sad': 0.30 + keyword_scores['sad'] * 0.20,
                'fearful': 0.10 + keyword_scores['fearful'] * 0.20
            }
        elif not has_positive_keywords and not has_negative_keywords and not has_excitement_keywords:
            # キーワードなし → 落ち着き優先（中立を排除）
            if is_positive:
                text_emotions = {
                    'happy': 0.12,
                    'joyful': 0.10,
                    'calm': 0.75,  # 落ち着きを大幅に高く
                    'excitement': 0.01,
                    'angry': 0.01,
                    'sad': 0.0,
                    'fearful': 0.01
                }
            else:
                # ネガティブだがキーワードがない場合は落ち着きと興奮に分散
                text_emotions = {
                    'happy': 0.05,
                    'joyful': 0.03,
                    'calm': 0.60,  # 落ち着きを高く
                    'excitement': 0.30,  # ネガティブな興奮（漠然とした不安定さ）
                    'angry': 0.01,
                    'sad': 0.01,
                    'fearful': 0.0
                }
        else:
            # 両方のキーワードがある場合（混在）
            if positive_keyword_count > negative_keyword_count:
                # ポジティブ優勢
                text_emotions = {
                    'happy': 0.22 + keyword_scores['happy'] * 0.15,
                    'joyful': 0.20 + keyword_scores['joyful'] * 0.15,
                    'calm': 0.25 + keyword_scores['calm'] * 0.15,
                    'excitement': 0.10 + keyword_scores['excitement'] * 0.15,
                    'angry': 0.09 + keyword_scores['angry'] * 0.10,
                    'sad': 0.10 + keyword_scores['sad'] * 0.10,
                    'fearful': 0.04 + keyword_scores['fearful'] * 0.10
                }
            else:
                # ネガティブ優勢
                text_emotions = {
                    'happy': 0.08 + keyword_scores['happy'] * 0.10,
                    'joyful': 0.07 + keyword_scores['joyful'] * 0.10,
                    'calm': 0.15 + keyword_scores['calm'] * 0.15,
                    'excitement': 0.15 + keyword_scores['excitement'] * 0.15,
                    'angry': 0.23 + keyword_scores['angry'] * 0.15,
                    'sad': 0.22 + keyword_scores['sad'] * 0.15,
                    'fearful': 0.10 + keyword_scores['fearful'] * 0.15
                }
        
        # 正規化
        total = sum(text_emotions.values())
        if total > 0:
            text_emotions = {k: v/total for k, v in text_emotions.items()}
        
        print(f"✓ テキスト感情分析完了")
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        text_emotions = None
else:
    print("  テキストが認識されなかったため、音声感情のみで判断します")
    text_emotions = None

# ========================================
# ステップ4: 統合分析
# ========================================
print("\n" + "=" * 70)
print("【ステップ4】統合分析結果")
print("=" * 70)

emotions_jp = {
    'happy': '幸福',
    'joyful': '喜び',
    'calm': '落ち着き',
    'excitement': '興奮',
    'angry': '怒り',
    'sad': '悲しみ',
    'fearful': '恐怖'
}

if text_emotions:
    # 音声とテキストの重み付け統合（テキストを重視: 60%）
    final_emotions = {}
    for emotion in voice_emotions.keys():
        final_emotions[emotion] = voice_emotions[emotion] * 0.4 + text_emotions[emotion] * 0.6
    
    print("\n📊 最終感情スコア（音声40% + テキスト60%）:\n")
else:
    # テキストがない場合は音声のみ
    final_emotions = voice_emotions
    print("\n📊 最終感情スコア（音声のみ）:\n")

# 固定順序で表示（幸福、喜び、落ち着き、興奮、怒り、悲しみ、恐怖）※中立を削除
emotion_order = ['happy', 'joyful', 'calm', 'excitement', 'angry', 'sad', 'fearful']

# 英語名の辞書
emotion_en = {
    'happy': 'happiness',
    'joyful': 'joy',
    'calm': 'calmness',
    'excitement': 'excitement',
    'angry': 'anger',
    'sad': 'sadness',
    'fearful': 'fear'
}

for emotion in emotion_order:
    prob = final_emotions[emotion]
    emotion_jp = emotions_jp[emotion]
    emotion_english = emotion_en[emotion]
    emotion_display = f"{emotion_jp}（{emotion_english}）"
    bar = "█" * int(prob * 50)
    
    # 音声とテキストの差分を表示
    if text_emotions:
        voice_val = voice_emotions[emotion]
        text_val = text_emotions[emotion]
        diff = abs(voice_val - text_val)
        if diff > 0.2:
            marker = " ⚠️ 不一致"
        else:
            marker = ""
        print(f"{emotion_display:20s}: {bar} {prob:.3f} (音声:{voice_val:.2f} / テキスト:{text_val:.2f}){marker}")
    else:
        print(f"{emotion_display:20s}: {bar} {prob:.3f}")

# 最も強い感情
dominant_emotion = max(final_emotions, key=final_emotions.get)
dominant_emotion_jp = emotions_jp[dominant_emotion]

print(f"\n🎯 判定結果: {dominant_emotion_jp} ({final_emotions[dominant_emotion]:.1%})")

# 心理的な距離感を計算 (1-10のスケール)
# 近い (10-8): ポジティブな感情が高い → 親密、安心
# 中間 (7-4): 混在または興奮
# 遠い (3-1): ネガティブな感情が高い → 警戒、距離を置く

positive_emotions = final_emotions['happy'] + final_emotions['joyful'] + final_emotions['calm']
negative_emotions = final_emotions['angry'] + final_emotions['sad'] + final_emotions['fearful']
excitement_level = final_emotions['excitement']

# 距離感の計算
# 基本値: ポジティブ感情が高いほど近い(10に近い)、ネガティブが高いほど遠い(1に近い)
base_distance = 5.5  # 中立からスタート
distance_score = base_distance + (positive_emotions * 4.5) - (negative_emotions * 4.5)

# 興奮は距離を少し遠くする(警戒しながらも関わる)
distance_score -= excitement_level * 1.5

# 1-10の範囲に収める
psychological_distance = max(1, min(10, int(round(distance_score))))

# 距離感の説明
if psychological_distance >= 8:
    distance_desc = "非常に近い（親密・安心）"
    distance_icon = "🤝"
elif psychological_distance >= 6:
    distance_desc = "やや近い（友好的）"
    distance_icon = "😊"
elif psychological_distance >= 4:
    distance_desc = "中立・やや遠い（慎重）"
    distance_icon = "🤔"
elif psychological_distance >= 2:
    distance_desc = "遠い（警戒・緊張）"
    distance_icon = "😰"
else:
    distance_desc = "非常に遠い（拒絶・回避）"
    distance_icon = "🚫"

print(f"\n📏 心理的な距離感: {psychological_distance}/10 {distance_icon}")
print(f"   → {distance_desc}")

# 統合判断の説明
if text_emotions:
    voice_dominant = max(voice_emotions, key=voice_emotions.get)
    text_dominant = max(text_emotions, key=text_emotions.get)
    
    print(f"\n💡 分析:")
    print(f"  音声からは「{emotions_jp[voice_dominant]}」の特徴")
    print(f"  言葉からは「{emotions_jp[text_dominant]}」の内容")
    
    if voice_dominant != text_dominant:
        print(f"  ⚠️  音声と言葉の感情が異なります")
        if voice_dominant in ['angry', 'sad', 'fearful'] and text_dominant in ['happy', 'joyful', 'calm']:
            print(f"  → ネガティブな気持ちを言葉で抑えている可能性")
        elif voice_dominant in ['happy', 'joyful', 'calm'] and text_dominant in ['angry', 'sad', 'fearful']:
            print(f"  → 表面的には穏やかだが内容は深刻")
    else:
        print(f"  ✅ 音声と言葉の感情が一致しています")

print("\n" + "=" * 70)
