import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np

# 録音設定
CHANNELS = 1  # モノラル
RATE = 44100  # サンプリングレート（Hz）
RECORD_SECONDS = 10  # 録音時間（秒）
OUTPUT_FILENAME = "output.wav"  # 保存するファイル名

print(f"{RECORD_SECONDS}秒間録音を開始します...")

# 録音
recording = sd.rec(int(RECORD_SECONDS * RATE), 
                   samplerate=RATE, 
                   channels=CHANNELS,
                   dtype='int16')

# 録音が完了するまで待機
sd.wait()

print("録音が完了しました。")

# wavファイルとして保存
wavfile.write(OUTPUT_FILENAME, RATE, recording)

print(f"{OUTPUT_FILENAME} として保存しました。")


