import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
from scipy.signal import butter, filtfilt
import wave
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import speech_recognition as sr
from transformers import pipeline
import os
import sys
import traceback

class EmotionAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("感情分析アプリケーション ✨")
        self.root.geometry("800x700")
        self.root.resizable(False, False)
        self.root.configure(bg="#FFF5F7")
        
        # ウィンドウが閉じられた時のハンドラを設定
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 録音設定
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 10
        self.OUTPUT_FILENAME = "output.wav"
        
        # モデルは必要になったときに読み込む
        self.model = None
        self.is_running = True
        self.feature_extractor = None
        self.sentiment_analyzer = None
        
        # GUI構築
        self.create_widgets()
        
    def create_widgets(self):
        """ウィジェットの作成"""
        # タイトル
        title_frame = tk.Frame(self.root, bg="#FFB6C1", height=70)
        title_frame.pack(fill=tk.X, pady=0)
        title_label = tk.Label(title_frame, text="🎤 感情分析アプリ ✨", 
                              font=("Meiryo UI", 22, "bold"), bg="#FFB6C1", fg="white")
        title_label.pack(pady=15)
        
        # ボタンフレーム（中央配置）
        button_frame = tk.Frame(self.root, bg="#FFF5F7", pady=30)
        button_frame.pack(fill=tk.X)
        
        # ボタンを中央に配置するための内部フレーム
        button_container = tk.Frame(button_frame, bg="#FFF5F7")
        button_container.pack(anchor=tk.CENTER)
        
        # 録音ボタン（水色）
        self.record_button = tk.Button(button_container, text="🎙️ 録音開始", 
                                       font=("Meiryo UI", 14, "bold"),
                                       bg="#87CEEB", fg="white",
                                       activebackground="#6FB8D9",
                                       width=18, height=2,
                                       relief=tk.FLAT,
                                       bd=0,
                                       cursor="hand2",
                                       command=self.start_recording)
        self.record_button.pack(side=tk.LEFT, padx=15)
        
        # ボタンにホバー効果を追加
        self.record_button.bind("<Enter>", lambda e: self.record_button.config(bg="#6FB8D9"))
        self.record_button.bind("<Leave>", lambda e: self.record_button.config(bg="#87CEEB"))
        
        # 分析ボタン（薄いオレンジ）
        self.analyze_button = tk.Button(button_container, text="📊 感情分析", 
                                        font=("Meiryo UI", 14, "bold"),
                                        bg="#FFB347", fg="white",
                                        activebackground="#FF9F2E",
                                        width=18, height=2,
                                        relief=tk.FLAT,
                                        bd=0,
                                        cursor="hand2",
                                        command=self.start_analysis)
        self.analyze_button.pack(side=tk.LEFT, padx=15)
        
        # ボタンにホバー効果を追加
        self.analyze_button.bind("<Enter>", lambda e: self.analyze_button.config(bg="#FF9F2E"))
        self.analyze_button.bind("<Leave>", lambda e: self.analyze_button.config(bg="#FFB347"))
        
        # ステータスラベル
        self.status_label = tk.Label(self.root, text="✨ 待機中... ✨", 
                                     font=("Meiryo UI", 12), fg="#FF69B4", bg="#FFF5F7")
        self.status_label.pack(pady=10)
        
        # 結果表示エリア
        result_frame = tk.Frame(self.root, bg="white")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        result_title = tk.Label(result_frame, text="📋 分析結果", 
                               font=("Meiryo UI", 14, "bold"), bg="white", fg="#FF69B4")
        result_title.pack(anchor=tk.W, pady=5)
        
        # スクロール可能なテキストエリア
        self.result_text = scrolledtext.ScrolledText(result_frame, 
                                                     font=("Meiryo UI", 10),
                                                     bg="#FFFAF0",
                                                     wrap=tk.WORD,
                                                     height=25)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
    def update_status(self, message, color="#666"):
        """ステータスラベルを更新"""
        try:
            if self.is_running:
                self.status_label.config(text=message, fg=color)
                self.root.update_idletasks()
        except Exception as e:
            print(f"ステータス更新エラー: {e}")
        
    def append_result(self, text):
        """結果テキストに追加"""
        try:
            if self.is_running:
                self.result_text.insert(tk.END, text + "\n")
                self.result_text.see(tk.END)
                self.root.update_idletasks()
        except Exception as e:
            print(f"結果表示エラー: {e}")
        
    def clear_result(self):
        """結果テキストをクリア"""
        self.result_text.delete(1.0, tk.END)
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        """バンドパスフィルターの設計"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def bandpass_filter(self, data, lowcut=80, highcut=3000, fs=44100, order=5):
        """バンドパスフィルターを適用"""
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    
    def normalize_audio(self, audio_data):
        """音量を正規化"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.9
        return audio_data
    
    def record_audio(self):
        """音声録音処理"""
        try:
            self.update_status(f"🎤 録音中... ({self.RECORD_SECONDS}秒)", "#dc3545")
            self.record_button.config(state=tk.DISABLED)
            self.analyze_button.config(state=tk.DISABLED)
            
            self.append_result("=" * 70)
            self.append_result("録音開始")
            self.append_result(f"録音時間: {self.RECORD_SECONDS}秒")
            self.append_result(f"サンプリングレート: {self.RATE} Hz")
            self.append_result("=" * 70)
            
            # 録音
            recording = sd.rec(int(self.RECORD_SECONDS * self.RATE), 
                             samplerate=self.RATE, 
                             channels=self.CHANNELS,
                             dtype='float32')
            sd.wait()
            
            if not self.is_running:
                return
            
            self.append_result("\n✓ 録音完了")
            self.append_result("🔧 音声処理中...")
            
            # 音声処理
            audio_data = recording.flatten()
            audio_filtered = self.bandpass_filter(audio_data, lowcut=80, highcut=3000, fs=self.RATE)
            audio_normalized = self.normalize_audio(audio_filtered)
            audio_int16 = np.int16(audio_normalized * 32767)
            
            # 保存
            wavfile.write(self.OUTPUT_FILENAME, self.RATE, audio_int16)
            
            self.append_result(f"✅ {self.OUTPUT_FILENAME} として保存しました")
            
            # 音声レベルチェック
            avg_amplitude = np.mean(np.abs(audio_int16))
            if avg_amplitude < 1000:
                self.append_result("⚠️  警告: 音声が小さすぎます")
            elif avg_amplitude > 20000:
                self.append_result("⚠️  警告: 音声が大きすぎます")
            else:
                self.append_result("✅ 音声レベルは適切です")
            
            self.update_status("録音完了！感情分析ボタンを押してください", "#28a745")
            
        except KeyboardInterrupt:
            self.append_result("\n⚠️  録音がキャンセルされました")
            self.update_status("録音キャンセル", "#ffc107")
        except Exception as e:
            error_msg = f"\n❌ 録音エラー: {str(e)}"
            self.append_result(error_msg)
            self.append_result(traceback.format_exc())
            self.update_status("録音エラー", "#dc3545")
            messagebox.showerror("録音エラー", f"録音中にエラーが発生しました:\n{str(e)}")
        finally:
            if self.is_running:
                self.record_button.config(state=tk.NORMAL)
                self.analyze_button.config(state=tk.NORMAL)
    
    def start_recording(self):
        """録音をスレッドで開始"""
        self.clear_result()
        thread = threading.Thread(target=self.record_audio)
        thread.daemon = True
        thread.start()
    
    def load_models(self):
        """モデルの読み込み（初回のみ）"""
        if self.model is None:
            try:
                self.append_result("\n🔄 AIモデル読み込み中...")
                self.update_status("モデル読み込み中...", "#ffc107")
                
                model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(device)
                self.model.eval()
                
                self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                                  model="distilbert-base-uncased-finetuned-sst-2-english")
                
                self.append_result("✓ モデル読み込み完了")
            except Exception as e:
                error_msg = f"モデル読み込みエラー: {str(e)}"
                self.append_result(f"\n❌ {error_msg}")
                self.append_result(traceback.format_exc())
                messagebox.showerror("モデルエラー", f"AIモデルの読み込みに失敗しました:\n{str(e)}\n\nインターネット接続を確認してください。")
                raise
    
    def analyze_emotion(self):
        """感情分析処理"""
        try:
            if not os.path.exists(self.OUTPUT_FILENAME):
                self.append_result("\n❌ エラー: output.wavファイルが見つかりません")
                self.append_result("先に録音ボタンを押してください")
                self.update_status("ファイルが見つかりません", "#dc3545")
                messagebox.showwarning("ファイルなし", "先に録音ボタンを押して音声を録音してください。")
                return
            
            if not self.is_running:
                return
            
            self.update_status("感情分析中...", "#ffc107")
            self.analyze_button.config(state=tk.DISABLED)
            self.record_button.config(state=tk.DISABLED)
            
            self.append_result("\n" + "=" * 70)
            self.append_result("【感情分析開始】")
            self.append_result("=" * 70)
            
            # モデル読み込み
            self.load_models()
            
            # ========================================
            # ステップ1: 音声感情分析
            # ========================================
            self.append_result("\n【ステップ1】音声感情分析")
            
            with wave.open(self.OUTPUT_FILENAME, 'r') as wav_file:
                num_channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                num_frames = wav_file.getnframes()
                
                buffer = wav_file.readframes(num_frames)
                
                if num_channels == 1:
                    audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_data = audio_data.reshape(-1, num_channels).mean(axis=1)
            
            # サンプリングレート変換
            if sample_rate != 16000:
                from scipy import signal
                audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
                sample_rate = 16000
            
            # 音声感情分析
            inputs = self.feature_extractor(audio_data, sampling_rate=sample_rate, 
                                           return_tensors="pt", padding=True)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                inputs = {key: val.to(device) for key, val in inputs.items()}
                logits = self.model(**inputs).logits
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
            
            neutral_prob = raw_results['neutral']
            neutral_to_calm = neutral_prob * 0.6
            neutral_to_excitement = neutral_prob * 0.4
            
            anger_prob = raw_results['angry'] + disgust_prob * 0.6
            excitement_from_anger = anger_prob * 0.5
            real_anger_prob = anger_prob * 0.5
            
            total_excitement = excitement_from_anger + neutral_to_excitement
            
            voice_emotions = {
                'happy': raw_results['happy'] * 0.3,
                'joyful': joyful_prob,
                'calm': raw_results['calm'] * 0.7 + neutral_to_calm,
                'excitement': total_excitement,
                'angry': real_anger_prob,
                'sad': raw_results['sad'],
                'fearful': raw_results['fearful'] + disgust_prob * 0.4
            }
            
            self.append_result("✓ 音声感情分析完了")
            
            # ========================================
            # ステップ2: 音声認識
            # ========================================
            self.append_result("\n【ステップ2】音声認識")
            
            recognizer = sr.Recognizer()
            text = ""
            
            try:
                with sr.AudioFile(self.OUTPUT_FILENAME) as source:
                    audio = recognizer.record(source)
                text = recognizer.recognize_google(audio, language="ja-JP")
                self.append_result(f"✓ 認識されたテキスト: 「{text}」")
            except sr.UnknownValueError:
                self.append_result("✗ 音声を認識できませんでした")
            except sr.RequestError as e:
                self.append_result(f"✗ エラー: {e}")
            
            # ========================================
            # ステップ3: テキスト感情分析
            # ========================================
            self.append_result("\n【ステップ3】テキスト感情分析")
            
            text_emotions = {}
            
            if text:
                # キーワードベースの感情検出
                keywords = {
                    'happy': ['嬉しい', '幸せ', '楽しい', '良い', 'いい', '最高', '素晴らしい', 'ありがとう', 'よかった', 'とても嬉しい', '大好き'],
                    'joyful': ['喜び', '喜ぶ', 'わくわく', 'ワクワク', '楽しみ', '面白い', 'うれしい', 'すごい', 'やった', 'わあ', 'やったー'],
                    'calm': ['落ち着', '穏やか', '平和', '安心', 'リラックス', '静か', 'ゆっくり'],
                    'excitement': ['イライラ', 'ソワソワ', '焦', '落ち着かない', 'バタバタ', '慌て', '急', '忙しい', '追われ'],
                    'angry': ['怒', '腹立', 'むかつ', '許せない', 'ふざけるな', 'うるさい', 'イラつ', 'ムカつ', 'やめろ', '馬鹿', 'バカ', 'ダメ', '最悪', 'ひどい', '酷い', '信じられない', '何', 'なに', 'うざ', 'ウザ', '黙れ', '切れ', 'キレ', '腹が立', '頭にくる', '頭に来る'],
                    'sad': ['悲しい', '寂しい', '辛い', '残念', '泣', '涙', '苦しい', '悲', 'どうして'],
                    'fearful': ['怖い', '不安', '心配', '恐ろしい', 'ドキドキ', '緊張', '震え', '嫌', '来ないで']
                }
                
                keyword_scores = {emotion: 0 for emotion in keywords.keys()}
                for emotion, words in keywords.items():
                    for word in words:
                        if word in text:
                            keyword_scores[emotion] += 1
                
                negative_keyword_count = keyword_scores['sad'] + keyword_scores['angry'] + keyword_scores['fearful']
                positive_keyword_count = keyword_scores['happy'] + keyword_scores['joyful']
                excitement_keyword_count = keyword_scores['excitement']
                
                has_positive_keywords = positive_keyword_count > 0
                has_negative_keywords = negative_keyword_count > 0
                has_excitement_keywords = excitement_keyword_count > 0
                
                # テキスト感情スコアの計算
                if has_positive_keywords and not has_negative_keywords and not has_excitement_keywords:
                    text_emotions = {
                        'happy': 0.30 + keyword_scores['happy'] * 0.20,
                        'joyful': 0.35 + keyword_scores['joyful'] * 0.20,
                        'calm': 0.35 + keyword_scores['calm'] * 0.20,
                        'excitement': 0.0,
                        'angry': 0.0,
                        'sad': 0.0,
                        'fearful': 0.0
                    }
                elif has_negative_keywords and not has_positive_keywords:
                    base_negative = 0.70 / max(negative_keyword_count, 1)
                    text_emotions = {
                        'happy': 0.0,
                        'joyful': 0.0,
                        'calm': 0.0,
                        'excitement': 0.15 + keyword_scores['excitement'] * 0.10,
                        'angry': base_negative + keyword_scores['angry'] * 0.15,
                        'sad': base_negative + keyword_scores['sad'] * 0.15,
                        'fearful': base_negative + keyword_scores['fearful'] * 0.15
                    }
                else:
                    # 混在またはキーワードなし
                    text_emotions = {
                        'happy': 0.15,
                        'joyful': 0.15,
                        'calm': 0.30,
                        'excitement': 0.15,
                        'angry': 0.10,
                        'sad': 0.10,
                        'fearful': 0.05
                    }
                
                # 正規化
                total = sum(text_emotions.values())
                if total > 0:
                    text_emotions = {k: v/total for k, v in text_emotions.items()}
                
                self.append_result("✓ テキスト感情分析完了")
            
            # ========================================
            # 最終統合
            # ========================================
            self.append_result("\n" + "=" * 70)
            self.append_result("【最終結果】")
            self.append_result("=" * 70)
            
            emotions_jp = {
                'happy': '幸福',
                'joyful': '喜び',
                'calm': '落ち着き',
                'excitement': '興奮',
                'angry': '怒り',
                'sad': '悲しみ',
                'fearful': '恐怖'
            }
            
            emotion_en = {
                'happy': 'happiness',
                'joyful': 'joy',
                'calm': 'calmness',
                'excitement': 'excitement',
                'angry': 'anger',
                'sad': 'sadness',
                'fearful': 'fear'
            }
            
            # 統合スコア計算
            final_emotions = {}
            if text_emotions:
                for emotion in voice_emotions.keys():
                    final_emotions[emotion] = voice_emotions[emotion] * 0.4 + text_emotions[emotion] * 0.6
                self.append_result("\n📊 最終感情スコア（音声40% + テキスト60%）:\n")
            else:
                final_emotions = voice_emotions
                self.append_result("\n📊 最終感情スコア（音声のみ）:\n")
            
            # 感情スコア表示
            emotion_order = ['happy', 'joyful', 'calm', 'excitement', 'angry', 'sad', 'fearful']
            
            for emotion in emotion_order:
                prob = final_emotions[emotion]
                emotion_jp = emotions_jp[emotion]
                emotion_english = emotion_en[emotion]
                emotion_display = f"{emotion_jp}（{emotion_english}）"
                bar = "█" * int(prob * 30)
                
                if text_emotions:
                    voice_val = voice_emotions[emotion]
                    text_val = text_emotions[emotion]
                    self.append_result(f"{emotion_display:20s}: {bar} {prob:.3f} (音声:{voice_val:.2f} / テキスト:{text_val:.2f})")
                else:
                    self.append_result(f"{emotion_display:20s}: {bar} {prob:.3f}")
            
            # 最も強い感情
            dominant_emotion = max(final_emotions, key=final_emotions.get)
            dominant_emotion_jp = emotions_jp[dominant_emotion]
            
            self.append_result(f"\n🎯 判定結果: {dominant_emotion_jp} ({final_emotions[dominant_emotion]:.1%})")
            
            # ========================================
            # 音声とテキストの判定結果比較
            # ========================================
            if text_emotions:
                voice_dominant = max(voice_emotions, key=voice_emotions.get)
                text_dominant = max(text_emotions, key=text_emotions.get)
                
                voice_dominant_jp = emotions_jp[voice_dominant]
                text_dominant_jp = emotions_jp[text_dominant]
                
                self.append_result(f"\n🔍 詳細分析:")
                self.append_result(f"  🎤 音声分析: {voice_dominant_jp} ({voice_emotions[voice_dominant]:.1%})")
                self.append_result(f"  💬 テキスト分析: {text_dominant_jp} ({text_emotions[text_dominant]:.1%})")
                
                # 音声とテキストの判定が異なる場合の推測
                if voice_dominant != text_dominant:
                    self.append_result(f"\n  ⚠️  音声と言葉の感情が異なります")
                    
                    # 具体的な推測を行う
                    insights = []
                    
                    # ネガティブな音声 + ポジティブなテキスト
                    if voice_dominant in ['angry', 'sad', 'fearful', 'excitement'] and text_dominant in ['happy', 'joyful', 'calm']:
                        if voice_dominant == 'angry':
                            insights.append("😤 内心では怒りを感じているが、言葉では抑えている可能性")
                            insights.append("   → 表面的には穏やかだが、本音は異なるかもしれません")
                        elif voice_dominant == 'sad':
                            insights.append("😢 悲しみを隠して明るく振る舞っている可能性")
                            insights.append("   → 無理をしている、または気を遣っているかもしれません")
                        elif voice_dominant == 'fearful':
                            insights.append("😰 不安や恐怖を感じながらも前向きな言葉を使っている")
                            insights.append("   → 心配事を抱えつつも、それを表に出さないようにしている")
                        elif voice_dominant == 'excitement':
                            insights.append("😣 焦りやイライラを感じながら、落ち着いた言葉を選んでいる")
                            insights.append("   → 不安定な気持ちを抑えようとしている可能性")
                    
                    # ポジティブな音声 + ネガティブなテキスト
                    elif voice_dominant in ['happy', 'joyful', 'calm'] and text_dominant in ['angry', 'sad', 'fearful', 'excitement']:
                        if text_dominant == 'angry':
                            insights.append("😠 穏やかな口調で怒りを表現している")
                            insights.append("   → 冷静に不満を伝えている、または言葉の裏に怒りがある可能性")
                        elif text_dominant == 'sad':
                            insights.append("😔 表面的には落ち着いているが、内容は深刻")
                            insights.append("   → 悲しい状況を冷静に受け止めようとしている")
                        elif text_dominant == 'fearful':
                            insights.append("😨 落ち着いた口調で不安や心配を語っている")
                            insights.append("   → 冷静さを保とうとしているが、内容は深刻な懸念を含む")
                        elif text_dominant == 'excitement':
                            insights.append("😖 穏やかに見えて、実は焦りや緊張を感じている")
                            insights.append("   → 言葉の裏に切迫感や不安定さが隠れている可能性")
                    
                    # 興奮が関与する特殊なケース
                    elif voice_dominant == 'excitement' or text_dominant == 'excitement':
                        insights.append("🌀 感情が不安定な状態")
                        insights.append("   → イライラ、焦り、または混乱した心理状態の可能性")
                    
                    # 幸福と喜びの違い
                    elif (voice_dominant == 'happy' and text_dominant == 'joyful') or (voice_dominant == 'joyful' and text_dominant == 'happy'):
                        insights.append("😊 ポジティブな感情で一致しています")
                        insights.append("   → 言葉と音声のトーンに若干の違いがありますが、全体的に良好")
                    
                    # その他の不一致
                    else:
                        insights.append("🤔 複雑な感情状態")
                        insights.append("   → 言葉の裏の意味がある可能性、または感情が混在している状態")
                    
                    # 推測を表示
                    self.append_result("")
                    self.append_result("  💡 推測される心理状態:")
                    for insight in insights:
                        self.append_result(f"  {insight}")
                    
                else:
                    self.append_result(f"\n  ✅ 音声と言葉の感情が一致しています")
                    self.append_result(f"     → 素直な感情表現、または一貫した心理状態")
            
            # ========================================
            # 心理的距離感の計算
            # ========================================
            positive_emotions = final_emotions['happy'] + final_emotions['joyful'] + final_emotions['calm']
            negative_emotions = final_emotions['angry'] + final_emotions['sad'] + final_emotions['fearful']
            excitement_level = final_emotions['excitement']
            
            base_distance = 5.5
            distance_score = base_distance + (positive_emotions * 4.5) - (negative_emotions * 4.5)
            distance_score -= excitement_level * 1.5
            
            psychological_distance = max(1, min(10, int(round(distance_score))))
            
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
            
            self.append_result(f"\n📏 心理的な距離感: {psychological_distance}/10 {distance_icon}")
            self.append_result(f"   → {distance_desc}")
            
            self.append_result("\n" + "=" * 70)
            self.append_result("分析完了！")
            self.append_result("=" * 70)
            
            self.update_status("分析完了！", "#28a745")
            
        except KeyboardInterrupt:
            self.append_result("\n⚠️  分析がキャンセルされました")
            self.update_status("分析キャンセル", "#ffc107")
        except Exception as e:
            error_msg = f"\n❌ 分析エラー: {str(e)}"
            self.append_result(error_msg)
            error_trace = traceback.format_exc()
            self.append_result(error_trace)
            self.update_status("分析エラー", "#dc3545")
            messagebox.showerror("分析エラー", f"感情分析中にエラーが発生しました:\n{str(e)}\n\n詳細はログを確認してください。")
        finally:
            if self.is_running:
                self.analyze_button.config(state=tk.NORMAL)
                self.record_button.config(state=tk.NORMAL)
    
    def start_analysis(self):
        """分析をスレッドで開始"""
        self.clear_result()
        thread = threading.Thread(target=self.analyze_emotion)
        thread.daemon = True
        thread.start()
    
    def on_closing(self):
        """ウィンドウを閉じる時の処理"""
        if messagebox.askokcancel("終了確認", "アプリケーションを終了しますか？"):
            self.is_running = False
            try:
                self.root.quit()
            except:
                pass
            try:
                self.root.destroy()
            except:
                pass

def main():
    try:
        root = tk.Tk()
        
        # アプリケーションのエラーハンドリング
        def report_callback_exception(exc_type, exc_value, exc_traceback):
            error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            print(f"予期しないエラー:\n{error_msg}")
            try:
                messagebox.showerror("エラー", f"予期しないエラーが発生しました:\n{exc_value}\n\nアプリケーションは継続します。")
            except:
                pass
        
        tk.Tk.report_callback_exception = report_callback_exception
        
        app = EmotionAnalyzerGUI(root)
        
        # メインループを堅牢に
        while True:
            try:
                root.mainloop()
                break
            except KeyboardInterrupt:
                print("キーボード割り込みを受信しました")
                break
            except Exception as e:
                print(f"メインループエラー: {e}")
                traceback.print_exc()
                # エラーが発生してもアプリを継続
                try:
                    root.update()
                except:
                    break
                    
    except Exception as e:
        print(f"致命的エラーが発生しました: {str(e)}")
        traceback.print_exc()
        try:
            messagebox.showerror("致命的エラー", f"アプリケーションの起動に失敗しました:\n{str(e)}")
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
