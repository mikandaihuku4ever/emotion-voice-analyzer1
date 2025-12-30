import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
from scipy.signal import butter, filtfilt

# 録音設定
CHANNELS = 1  # モノラル
RATE = 44100  # サンプリングレート（Hz）
RECORD_SECONDS = 10  # 録音時間（秒）
OUTPUT_FILENAME = "output.wav"  # 保存するファイル名

def butter_bandpass(lowcut, highcut, fs, order=5):
    """バンドパスフィルターの設計"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=80, highcut=3000, fs=44100, order=5):
    """バンドパスフィルターを適用（人間の声の周波数範囲）"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def normalize_audio(audio_data):
    """音量を正規化"""
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val * 0.9  # 90%に正規化
    return audio_data

print("=" * 50)
print("【改善版】音声録音プログラム")
print("=" * 50)
print(f"\n📌 録音設定:")
print(f"  - サンプリングレート: {RATE} Hz")
print(f"  - 録音時間: {RECORD_SECONDS} 秒")
print(f"  - 保存先: {OUTPUT_FILENAME}")
print("\n💡 録音のコツ:")
print("  - マイクから 15-30cm の距離を保つ")
print("  - 静かな環境で録音する")
print("  - はっきりと話す")
print("  - 感情を込めて話す（自然な感情表現）")

input("\n準備ができたらEnterキーを押してください...")

print(f"\n🎤 {RECORD_SECONDS}秒間録音を開始します...\n")

# 録音
recording = sd.rec(int(RECORD_SECONDS * RATE), 
                   samplerate=RATE, 
                   channels=CHANNELS,
                   dtype='float32')  # float32で録音（精度向上）

# 録音が完了するまで待機
sd.wait()

print("✓ 録音が完了しました。")
print("\n🔧 音声処理中...")

# float32からnumpy配列へ
audio_data = recording.flatten()

# 1. ノイズ除去：バンドパスフィルター適用
print("  - バンドパスフィルター適用（80-3000 Hz）")
audio_filtered = bandpass_filter(audio_data, lowcut=80, highcut=3000, fs=RATE)

# 2. 音量正規化
print("  - 音量正規化")
audio_normalized = normalize_audio(audio_filtered)

# 3. int16に変換（WAVファイル用）
audio_int16 = np.int16(audio_normalized * 32767)

# wavファイルとして保存
wavfile.write(OUTPUT_FILENAME, RATE, audio_int16)

print(f"\n✅ {OUTPUT_FILENAME} として保存しました。")
print("\n📊 音声情報:")
print(f"  - 長さ: {len(audio_int16) / RATE:.2f} 秒")
print(f"  - サンプル数: {len(audio_int16)}")
print(f"  - 最大振幅: {np.max(np.abs(audio_int16))}")

# 音声レベルのチェック
avg_amplitude = np.mean(np.abs(audio_int16))
if avg_amplitude < 1000:
    print("\n⚠️  警告: 音声が小さすぎます。マイクを近づけるか音量を上げてください。")
elif avg_amplitude > 20000:
    print("\n⚠️  警告: 音声が大きすぎます。マイクを遠ざけるか音量を下げてください。")
else:
    print("\n✅ 音声レベルは適切です。")

print("=" * 50)
