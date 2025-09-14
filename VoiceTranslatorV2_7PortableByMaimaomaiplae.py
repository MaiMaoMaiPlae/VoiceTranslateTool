import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import threading
import queue
import time
import datetime
import librosa
import numpy as np
import soundfile as sf
import configparser
import re
import json
import pickle
import subprocess
import sys
from pydub import AudioSegment
from googletrans import Translator, LANGUAGES
import tempfile
import urllib.request
import urllib.error
import shutil
from enum import Enum, auto # เพิ่ม auto จาก enum
# ===============================================
# RESOURCE PATH MANAGEMENT
# ===============================================
def get_resource_path(relative_path):
   try:
       base_path = sys._MEIPASS  # PyInstaller temp folder
   except Exception:
       base_path = os.path.abspath(".")
   return os.path.join(base_path, relative_path)
# ===============================================
# FFMPEG SETUP
# ===============================================
def setup_ffmpeg():
   ffmpeg_path = get_resource_path("ffmpeg/bin")
   if os.path.exists(ffmpeg_path):
       os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
# ===============================================
# WHISPER ENVIRONMENT SETUP
# ===============================================
def create_whisper_safe_environment():
   """Create a safe environment for Whisper on fresh Windows"""
   try:
       temp_dir = tempfile.gettempdir()
       whisper_cache = os.path.join(temp_dir, "whisper_models")
       torch_cache = os.path.join(temp_dir, "torch_cache")
       hf_cache = os.path.join(temp_dir, "huggingface_cache")
       os.makedirs(whisper_cache, exist_ok=True)
       os.makedirs(torch_cache, exist_ok=True) 
       os.makedirs(hf_cache, exist_ok=True)
       os.environ['WHISPER_CACHE_DIR'] = whisper_cache
       os.environ['TORCH_HOME'] = torch_cache
       os.environ['HF_HOME'] = hf_cache
       os.environ['TRANSFORMERS_CACHE'] = hf_cache
       os.environ['TOKENIZERS_PARALLELISM'] = 'false'
       os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
       os.environ['TEMP'] = temp_dir
       os.environ['TMP'] = temp_dir
       print(f"Whisper environment created: {whisper_cache}")
       return True
   except Exception as e:
       print(f"Failed to create Whisper environment: {e}")
       return False
def safe_whisper_import():
   """Safely import Whisper with full error handling"""
   try:
       create_whisper_safe_environment()
       import torch
       print("PyTorch imported successfully")
       torch.set_num_threads(1)
       import whisper
       print("Whisper imported successfully")
       if not hasattr(whisper, 'load_model'):
           raise AttributeError("Whisper module incomplete - missing load_model")
       if not hasattr(whisper, 'available_models'):
           raise AttributeError("Whisper module incomplete - missing available_models")
       return True, whisper
   except ImportError as e:
       print(f"Whisper import failed: {e}")
       return False, None
   except Exception as e:
       print(f"Whisper setup failed: {e}")
       return False, None
def safe_whisper_load_model_from_file(model_path, log_widget=None):
   """Safely load Whisper model from local file"""
   def log_msg(msg, level="INFO"):
       print(f"[{level}] {msg}")
       if log_widget:
           log_message(log_widget, msg, level=level)
   try:
       log_msg(f"Starting Whisper model loading from file: {model_path}")
       if not os.path.exists(model_path):
           log_msg(f"Model file not found: {model_path}", "ERROR")
           return None
       if not create_whisper_safe_environment():
           log_msg("Failed to create Whisper environment", "ERROR")
           return None
       whisper_available, whisper = safe_whisper_import()
       if not whisper_available:
           log_msg("Whisper import failed", "ERROR")
           return None
       import torch
       device = "cpu"
       log_msg(f"Using device: {device}")
       log_msg(f"Loading Whisper model from: {model_path}")
       try:
           with torch.no_grad():
               model = whisper.load_model(model_path, device=device)
           log_msg(f"Model loaded successfully from {model_path}", "INFO")
           return model
       except (FileNotFoundError, PermissionError, RuntimeError) as e:
           log_msg(f"Error loading model: {e}", "ERROR")
           return None
       except Exception as e:
           log_msg(f"Unexpected error: {e}", "ERROR")
           return None
   except Exception as critical_e:
       log_msg(f"Critical error in Whisper setup: {critical_e}", "ERROR")
       return None
def safe_whisper_load_model(model_name, log_widget=None):
   """Safely load Whisper model with comprehensive error handling"""
   def log_msg(msg, level="INFO"):
       print(f"[{level}] {msg}")
       if log_widget:
           log_message(log_widget, msg, level=level)
   try:
       log_msg(f"Starting Whisper model loading: {model_name}")
       try:
           urllib.request.urlopen('https://www.google.com', timeout=10)
           log_msg("Internet connection verified")
       except:
           log_msg("No internet connection - cannot download models", "ERROR")
           return None
       if not create_whisper_safe_environment():
           log_msg("Failed to create Whisper environment", "ERROR")
           return None
       whisper_available, whisper = safe_whisper_import()
       if not whisper_available:
           log_msg("Whisper import failed", "ERROR")
           return None
       import torch
       device = "cpu"
       log_msg(f"Using device: {device}")
       cache_dir = os.environ.get('WHISPER_CACHE_DIR')
       log_msg(f"Cache directory: {cache_dir}")
       log_msg(f"Loading Whisper model: {model_name}")
       try:
           with torch.no_grad():
               model = whisper.load_model(
                   model_name,
                   device=device,
                   download_root=cache_dir,
                   in_memory=False
               )
           log_msg(f"Model {model_name} loaded successfully", "INFO")
           return model
       except (urllib.error.URLError, urllib.error.HTTPError, FileNotFoundError, PermissionError, RuntimeError) as e:
           log_msg(f"Error loading model: {e}", "ERROR")
           # Fallback to tiny model
           if model_name != "tiny":
               log_msg("Trying fallback to 'tiny' model...", "WARNING")
               try:
                   with torch.no_grad():
                       model = whisper.load_model("tiny", device="cpu", download_root=cache_dir)
                   log_msg("Fallback to tiny model successful", "INFO")
                   return model
               except Exception as fallback_e:
                   log_msg(f"Fallback failed: {fallback_e}", "ERROR")
           return None
       except Exception as e:
           log_msg(f"Unexpected error: {e}", "ERROR")
           return None
   except Exception as critical_e:
       log_msg(f"Critical error in Whisper setup: {critical_e}", "ERROR")
       return None
# ===============================================
# LIBRARY AVAILABILITY CHECK
# ===============================================
# Setup FFmpeg
setup_ffmpeg()
# Initialize libraries with error handling
WHISPER_AVAILABLE = False
try:
   whisper_ok, _ = safe_whisper_import()
   WHISPER_AVAILABLE = whisper_ok
   if WHISPER_AVAILABLE:
       print("Whisper is available")
   else:
       print("Warning: Whisper library not available")
except Exception as e:
   print(f"Warning: Whisper initialization failed: {e}")
try:
   from google.cloud import speech
   from google.api_core.exceptions import GoogleAPICallError, NotFound, PermissionDenied
   GOOGLE_STT_AVAILABLE = True
except ImportError:
   GOOGLE_STT_AVAILABLE = False
   print("Warning: google-cloud-speech library not found")
try:
   import google.generativeai as genai
   GEMINI_AVAILABLE = True
except ImportError:
   GEMINI_AVAILABLE = False
   print("Warning: google-generativeai library not found")
try:
   from google.cloud import texttospeech
   GOOGLE_TTS_AVAILABLE = True
except ImportError:
   GOOGLE_TTS_AVAILABLE = False
   print("Warning: google-cloud-texttospeech library not found")
try:
   from gtts import gTTS
   GTTS_AVAILABLE = True
except ImportError:
   GTTS_AVAILABLE = False
   print("Warning: gTTS library not found")
# ===============================================
# APP CONFIGURATION
# ===============================================
APP_NAME = "Voice Translator Tools By Maimaomaiplae"
APP_VERSION = "2.0"
WINDOW_SIZE = "1200x850"
THEME = "dark-blue"
FONT_NAME = "TkDefaultFont"
FONT_SIZE = 16
DEFAULT_FONT = (FONT_NAME, FONT_SIZE)
SUPPORTED_AUDIO_EXTENSIONS = ['.ogg', '.wav', '.mp3', '.flac', '.m4a', '.aac']
SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.wmv', '.mkv', '.mpg', '.flv', '.mov']
SUPPORTED_INPUT_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS
OUTPUT_ANALYSIS_FILE = "Analyzed_Sounds.txt"
OUTPUT_TRANSCRIPTION_FILE = "OriginalSounds.txt"
OUTPUT_TRANSLATION_FILE = "Translated_Sounds.txt"
LOG_FILE = "Log.txt"
DEFAULT_VOICE_FILE = "Select Input Voice file (.txt)"
DEFAULT_GEMINI_SETTINGS_FILE = "Geminikey.ini"
DEFAULT_COPILOT_SETTINGS_FILE = "Copilotkey.ini"
EMOTION_ENHANCED_CONFIG_FILE = "emotion_enhanced_config.json"
# Thai emotion-compatible voices configuration
THAI_EMOTION_COMPATIBILITY = {
    "compatible_voices": [
        "th-TH-Neural2-C",
        "th-TH-Standard-A"
    ],
    "voice_features": {
        "th-TH-Neural2-C": {
            "supports_ssml": True,
            "supports_prosody": True,
            "supports_emotion": True,
            "voice_type": "neural2",
            "language": "th-TH",
            "gender": "female"
        },
        "th-TH-Standard-A": {
            "supports_ssml": True,
            "supports_prosody": True,
            "supports_emotion": True,
            "voice_type": "standard",
            "language": "th-TH",
            "gender": "female"
        }
    },
    "emotion_mapping": {
        "neutral": {"rate": "1.0", "pitch": "0st", "volume": "medium"},
        "sad": {"rate": "0.75", "pitch": "-3st", "volume": "soft"},
        "happy": {"rate": "1.2", "pitch": "+2st", "volume": "loud"},
        "angry": {"rate": "1.1", "pitch": "+3st", "volume": "loud"},
        "excited": {"rate": "1.3", "pitch": "+4st", "volume": "x-loud"},
        "calm": {"rate": "0.85", "pitch": "-1st", "volume": "medium"},
        "surprise": {"rate": "1.15", "pitch": "+1st", "volume": "medium"},
        "whisper": {"rate": "0.6", "pitch": "-2st", "volume": "x-soft"}
    }
}
DEFAULT_CHATGPT_SETTINGS_FILE = "ChatGPTkey.ini"
SETTINGS_FILE = "app_settings.ini"
EMOTION_CONFIG_FILE = "emotion_config.json"
EMOTION_LOG_FILE = "emotion_analysis_log.txt"
# Progress tracking files
PROGRESS_DIR = "progress_tracking"
FREQUENCY_PROGRESS_FILE = "frequency_progress.json"
TRANSCRIPTION_PROGRESS_FILE = "transcription_progress.json"
TRANSLATION_PROGRESS_FILE = "translation_progress.json"
SYNTHESIS_PROGRESS_FILE = "synthesis_progress.json"
# เพิ่มใหม่: Time tracking files
TIME_TRACKING_FILE = "time_tracking.json"
# เพิ่มใหม่: Gender frequency range settings
DEFAULT_MALE_MIN_HZ = 00
DEFAULT_MALE_MAX_HZ = 90
DEFAULT_FEMALE_MIN_HZ = 91
DEFAULT_FEMALE_MAX_HZ = 5000
# Global variables
input_folder_path = ""
output_folder_path = ""
processing_queue = queue.Queue()
stop_processing_flag = threading.Event()
gender_analysis_results = {}
# เพิ่มใหม่: Global variables for time tracking
current_task_start_time = None
accumulated_task_times = {}
# ===============================================
# EMOTION ANALYSIS SYSTEM
# ===============================================
class EmotionAnalyzer:
   """คลาสสำหรับวิเคราะห์อารมณ์จากข้อความ"""
   def __init__(self, config_file=EMOTION_CONFIG_FILE):
       self.config_file = config_file
       self.emotion_config = self.load_emotion_config()
   def load_emotion_config(self):
       """โหลดค่าตั้งค่าอารมณ์จากไฟล์"""
       default_config = {
           "neutral": {
               "keywords": ["ปกติ", "ธรรมดา", "เหมือนเดิม"],
               "ssml": {"rate": "1.0", "pitch": "0st", "volume": "medium"}
           },
           "sad": {
               "keywords": ["เศร้า", "เสียใจ", "หดหู่", "ท้อแท้", "หนาว", "เหงา", "ผิดหวัง"],
               "ssml": {"rate": "0.75", "pitch": "-3st", "volume": "soft"}
           },
           "happy": {
               "keywords": ["ดีใจ", "มีความสุข", "ยินดี", "สนุก", "ร่าเริง", "สุขใจ"],
               "ssml": {"rate": "1.2", "pitch": "+2st", "volume": "loud"}
           },
           "angry": {
               "keywords": ["โกรธ", "หงุดหงิด", "ไม่", "เลิก", "หยุด", "อย่า"],
               "ssml": {"rate": "1.1", "pitch": "+3st", "volume": "loud"}
           },
           "excited": {
               "keywords": ["ตื่นเต้น", "เร้าใจ", "ฮือฮา", "อิดโรย", "วาว"],
               "ssml": {"rate": "1.3", "pitch": "+4st", "volume": "x-loud"}
           },
           "calm": {
               "keywords": ["สงบ", "เงียบ", "ผ่อนคลาย", "สบายใจ", "นิ่ง"],
               "ssml": {"rate": "0.85", "pitch": "-1st", "volume": "medium"}
           },
           "surprise": {
               "keywords": ["แปลกใจ", "ประหลาดใจ", "งง", "สงสัย", "อ้าว", "อะไร"],
               "ssml": {"rate": "1.15", "pitch": "+1st", "volume": "medium"}
           },
           "whisper": {
               "keywords": ["กระซิบ", "เบาๆ", "เงียบๆ", "แอบ"],
               "ssml": {"rate": "0.6", "pitch": "-2st", "volume": "x-soft"}
           }
       }
       try:
           if os.path.exists(self.config_file):
               with open(self.config_file, 'r', encoding='utf-8') as f:
                   loaded_config = json.load(f)
               # ผสมกับค่าเริ่มต้น
               for emotion, data in default_config.items():
                   if emotion not in loaded_config:
                       loaded_config[emotion] = data
               return loaded_config
           else:
               self.save_emotion_config(default_config)
               return default_config
       except Exception as e:
           print(f"Error loading emotion config: {e}")
           return default_config
   def save_emotion_config(self, config=None):
       """บันทึกค่าตั้งค่าอารมณ์ลงไฟล์"""
       if config is None:
           config = self.emotion_config
       try:
           with open(self.config_file, 'w', encoding='utf-8') as f:
               json.dump(config, f, ensure_ascii=False, indent=2)
           return True
       except Exception as e:
           print(f"Error saving emotion config: {e}")
           return False
   def analyze_simple(self, text):
       """วิเคราะห์อารมณ์แบบง่าย (ทั้งประโยค) - ใช้คำแรกที่พบ"""
       if not text or not isinstance(text, str):
           return {
               "emotion": "neutral",
               "confidence": 0,
               "first_keyword": None,
               "first_position": -1,
               "all_scores": {},
               "text": text or ""
           }
       text_lower = text.lower()
       first_emotion = None
       first_position = len(text)
       first_keyword = None
       # หาคำอารมณ์แรกที่ปรากฏในประโยค
       for emotion, data in self.emotion_config.items():
           keywords = data.get('keywords', [])
           for keyword in keywords:
               keyword_lower = keyword.lower()
               position = text_lower.find(keyword_lower)
               if position != -1 and position < first_position:
                   first_position = position
                   first_emotion = emotion
                   first_keyword = keyword
       # ถ้าไม่พบคำอารมณ์ใด ให้เป็น neutral
       if first_emotion is None:
           first_emotion = "neutral"
           confidence = 0
       else:
           confidence = 1
       # คำนวณคะแนนทั้งหมด (สำหรับ backward compatibility)
       emotion_scores = {}
       for emotion, data in self.emotion_config.items():
           score = 0
           keywords = data.get('keywords', [])
           for keyword in keywords:
               if keyword.lower() in text_lower:
                   score += text_lower.count(keyword.lower())
           emotion_scores[emotion] = score
       return {
           "emotion": first_emotion,
           "confidence": confidence,
           "first_keyword": first_keyword,
           "first_position": first_position,
           "all_scores": emotion_scores,
           "text": text
       }
   def analyze_advanced(self, text):
       """วิเคราะห์อารมณ์แบบขั้นสูง (แบ่งประโยคย่อยตามช่องว่าง)"""
       if not text or not isinstance(text, str):
           return {
               "mode": "advanced",
               "sentences": [self.analyze_simple("")],
               "full_text": text or "",
               "sentence_count": 1
           }
       # แบ่งประโยคด้วยช่องว่าง (whitespace) สำหรับภาษาไทย
       sentences = re.split(r'\s+', text.strip())
       # รวมคำที่อยู่ติดกันจนกว่าจะเจอคำอารมณ์หรือจบประโยค
       processed_sentences = []
       current_sentence = ""
       for word in sentences:
           if not word.strip():
               continue
           # เพิ่มคำเข้าไปในประโยคปัจจุบัน
           if current_sentence:
               test_sentence = current_sentence + " " + word
           else:
               test_sentence = word
           # ตรวจสอบว่ามีคำอารมณ์ในประโยคนี้หรือไม่
           has_emotion = self._has_emotion_keyword(test_sentence)
           if has_emotion:
               # ถ้ามีคำอารมณ์ ให้จบประโยคที่นี่
               processed_sentences.append(test_sentence.strip())
               current_sentence = ""
           else:
               # ถ้ายังไม่มีคำอารมณ์ ให้เก็บไว้
               current_sentence = test_sentence
       # เพิ่มประโยคสุดท้ายถ้ายังเหลืออยู่
       if current_sentence.strip():
           processed_sentences.append(current_sentence.strip())
       # วิเคราะห์อารมณ์ของแต่ละประโยคย่อย
       results = []
       for sentence in processed_sentences:
           if sentence.strip():
               result = self.analyze_simple(sentence)
               results.append(result)
       # ถ้าไม่มีประโยคย่อย ให้ใช้ข้อความทั้งหมด
       if not results:
           results = [self.analyze_simple(text)]
       return {
           "mode": "advanced",
           "sentences": results,
           "full_text": text,
           "sentence_count": len(results)
       }
   def _has_emotion_keyword(self, text):
       """ตรวจสอบว่าข้อความมีคำศัพท์อารมณ์หรือไม่"""
       if not text or not isinstance(text, str):
           return False
       text_lower = text.lower()
       for emotion, data in self.emotion_config.items():
           keywords = data.get('keywords', [])
           for keyword in keywords:
               if keyword.lower() in text_lower:
                   return True
       return False
   def find_first_emotion_keyword(self, text):
       """หาคำอารมณ์แรกที่ปรากฏในข้อความ"""
       if not text or not isinstance(text, str):
           return {
               "emotion": None,
               "keyword": None,
               "position": -1
           }
       text_lower = text.lower()
       first_emotion = None
       first_position = len(text)
       first_keyword = None
       for emotion, data in self.emotion_config.items():
           keywords = data.get('keywords', [])
           for keyword in keywords:
               keyword_lower = keyword.lower()
               position = text_lower.find(keyword_lower)
               if position != -1 and position < first_position:
                   first_position = position
                   first_emotion = emotion
                   first_keyword = keyword
       return {
           "emotion": first_emotion,
           "keyword": first_keyword,
           "position": first_position if first_emotion else -1
       }
   def get_emotion_keywords_flat(self):
       """ได้รับคำศัพท์อารมณ์ทั้งหมดในรูปแบบ flat list"""
       all_keywords = []
       for emotion, data in self.emotion_config.items():
           keywords = data.get('keywords', [])
           for keyword in keywords:
               all_keywords.append({
                   'keyword': keyword,
                   'emotion': emotion
               })
       return all_keywords
   def get_emotion_keywords(self, emotion):
       """ได้รับคำสำคัญของอารมณ์"""
       return self.emotion_config.get(emotion, {}).get('keywords', [])
   def add_emotion_keyword(self, emotion, keyword):
       """เพิ่มคำสำคัญให้อารมณ์"""
       if emotion in self.emotion_config:
           if keyword not in self.emotion_config[emotion]['keywords']:
               self.emotion_config[emotion]['keywords'].append(keyword)
               return self.save_emotion_config()
       return False
   def remove_emotion_keyword(self, emotion, keyword):
       """ลบคำสำคัญจากอารมณ์"""
       if emotion in self.emotion_config:
           if keyword in self.emotion_config[emotion]['keywords']:
               self.emotion_config[emotion]['keywords'].remove(keyword)
               return self.save_emotion_config()
       return False
class SSMLGenerator:
   """คลาสสำหรับสร้าง SSML จากผลการวิเคราะห์อารมณ์"""
   def __init__(self, emotion_analyzer):
       self.emotion_analyzer = emotion_analyzer
   def create_simple_ssml(self, analysis_result):
       """สร้าง SSML แบบง่าย (ทั้งประโยค) - ใช้อารมณ์แรกที่พบ"""
       if not analysis_result or not isinstance(analysis_result, dict):
           return "<speak>Invalid analysis result</speak>"
       emotion = analysis_result.get('emotion', 'neutral')
       text = analysis_result.get('text', '')
       if not text:
           return "<speak></speak>"
       ssml_config = self.emotion_analyzer.emotion_config.get(emotion, {}).get('ssml', {})
       rate = ssml_config.get('rate', '1.0')
       pitch = ssml_config.get('pitch', '0st')
       volume = ssml_config.get('volume', 'medium')
       # แสดงข้อมูลการวิเคราะห์ถ้ามีคำอารมณ์
       debug_info = ""
       if analysis_result.get('first_keyword'):
           debug_info = f"<!-- พบคำ '{analysis_result['first_keyword']}' อารมณ์ '{emotion}' -->\n"
       ssml = f'''<speak>
{debug_info}<prosody rate="{rate}" pitch="{pitch}" volume="{volume}">
{text}
</prosody>
</speak>'''
       return ssml.strip()
   def create_advanced_ssml(self, analysis_result):
       """สร้าง SSML แบบขั้นสูง (แบ่งประโยคย่อยตามอารมณ์)"""
       if not analysis_result or not isinstance(analysis_result, dict):
           return "<speak>Invalid analysis result</speak>"
       if analysis_result.get('mode') != 'advanced':
           return self.create_simple_ssml(analysis_result)
       sentences = analysis_result.get('sentences', [])
       if not sentences:
           return "<speak></speak>"
       ssml_parts = ['<speak>']
       for i, sentence_result in enumerate(sentences):
           emotion = sentence_result.get('emotion', 'neutral')
           text = sentence_result.get('text', '')
           if not text:
               continue
           ssml_config = self.emotion_analyzer.emotion_config.get(emotion, {}).get('ssml', {})
           rate = ssml_config.get('rate', '1.0')
           pitch = ssml_config.get('pitch', '0st')
           volume = ssml_config.get('volume', 'medium')
           # แสดงข้อมูลการวิเคราะห์ถ้ามีคำอารมณ์
           debug_info = ""
           if sentence_result.get('first_keyword'):
               debug_info = f"<!-- ส่วนที่ {i+1}: พบคำ '{sentence_result['first_keyword']}' อารมณ์ '{emotion}' -->\n"
           ssml_parts.append(f'{debug_info}<prosody rate="{rate}" pitch="{pitch}" volume="{volume}">{text}</prosody>')
           # เพิ่มช่วงหยุดระหว่างประโยค (ยกเว้นประโยคสุดท้าย)
           if i < len(sentences) - 1:
               ssml_parts.append('<break time="300ms"/>')
       ssml_parts.append('</speak>')
       return '\n'.join(ssml_parts)
   def test_ssml_generation(self, text, mode='simple'):
       """ทดสอบการสร้าง SSML พร้อมข้อมูลการวิเคราะห์"""
       if not text or not isinstance(text, str):
           empty_analysis = {
               'emotion': 'neutral',
               'confidence': 0,
               'first_keyword': None,
               'text': text or ''
           }
           return "<speak></speak>", empty_analysis
       if mode == 'simple':
           analysis = self.emotion_analyzer.analyze_simple(text)
           ssml = self.create_simple_ssml(analysis)
           # เพิ่มข้อมูลสำหรับการแสดงผล
           analysis['analysis_info'] = {
               'mode': 'simple',
               'found_keywords': [analysis.get('first_keyword')] if analysis.get('first_keyword') else [],
               'emotion_distribution': {analysis['emotion']: 1}
           }
           return ssml, analysis
       else:
           analysis = self.emotion_analyzer.analyze_advanced(text)
           ssml = self.create_advanced_ssml(analysis)
           # เพิ่มข้อมูลสำหรับการแสดงผล
           found_keywords = []
           emotion_distribution = {}
           for sentence_result in analysis.get('sentences', []):
               if sentence_result.get('first_keyword'):
                   found_keywords.append(sentence_result['first_keyword'])
               emotion = sentence_result.get('emotion', 'neutral')
               emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
           analysis['analysis_info'] = {
               'mode': 'advanced',
               'found_keywords': found_keywords,
               'emotion_distribution': emotion_distribution
           }
           return ssml, analysis
# ===============================================
# EMOTION MANAGEMENT UI
# ===============================================
class EmotionManagerWindow:
   """หน้าต่างสำหรับจัดการคำศัพท์อารมณ์"""
   def __init__(self, parent, emotion_analyzer):
       self.parent = parent
       self.emotion_analyzer = emotion_analyzer
       self.window = None
       self.create_window()
   def create_window(self):
       """สร้างหน้าต่างจัดการอารมณ์"""
       self.window = ctk.CTkToplevel(self.parent)
       self.window.title("จัดการคำศัพท์อารมณ์")
       self.window.geometry("900x700")
       self.window.transient(self.parent)
       self.window.grab_set()
       # Main frame
       main_frame = ctk.CTkFrame(self.window)
       main_frame.pack(fill="both", expand=True, padx=10, pady=10)
       # Title
       title_label = ctk.CTkLabel(main_frame, text="จัดการคำศัพท์อารมณ์", 
                                 font=ctk.CTkFont(size=20, weight="bold"))
       title_label.pack(pady=10)
       # Emotion selection
       emotion_frame = ctk.CTkFrame(main_frame)
       emotion_frame.pack(fill="x", padx=10, pady=5)
       ctk.CTkLabel(emotion_frame, text="เลือกอารมณ์:").pack(side="left", padx=5)
       self.emotion_var = tk.StringVar(value="neutral")
       emotions = list(self.emotion_analyzer.emotion_config.keys())
       self.emotion_menu = ctk.CTkOptionMenu(emotion_frame, variable=self.emotion_var, 
                                            values=emotions, command=self.load_emotion_keywords)
       self.emotion_menu.pack(side="left", padx=5)
       # Test area
       test_frame = ctk.CTkFrame(main_frame)
       test_frame.pack(fill="x", padx=10, pady=5)
       ctk.CTkLabel(test_frame, text="ทดสอบข้อความ:").pack(side="left", padx=5)
       self.test_entry = ctk.CTkEntry(test_frame, width=300, placeholder_text="พิมพ์ข้อความเพื่อทดสอบ...")
       self.test_entry.pack(side="left", padx=5)
       ctk.CTkButton(test_frame, text="ทดสอบ", command=self.test_emotion).pack(side="left", padx=5)
       # Keywords management
       keywords_frame = ctk.CTkFrame(main_frame)
       keywords_frame.pack(fill="both", expand=True, padx=10, pady=5)
       # Left side - current keywords
       left_frame = ctk.CTkFrame(keywords_frame)
       left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
       ctk.CTkLabel(left_frame, text="คำศัพท์ปัจจุบัน (1 คำต่อบรรทัด):", font=ctk.CTkFont(weight="bold")).pack(pady=5)
       self.keywords_textbox = ctk.CTkTextbox(left_frame, height=300, width=300)
       self.keywords_textbox.pack(fill="both", expand=True, padx=5, pady=5)
       # Right side - add/remove keywords
       right_frame = ctk.CTkFrame(keywords_frame)
       right_frame.pack(side="right", fill="y", padx=5, pady=5)
       ctk.CTkLabel(right_frame, text="จัดการคำศัพท์:", font=ctk.CTkFont(weight="bold")).pack(pady=5)
       # Add single keyword
       ctk.CTkLabel(right_frame, text="เพิ่มคำศัพท์เดียว:", font=ctk.CTkFont(size=12)).pack(pady=(10,2))
       self.new_keyword_entry = ctk.CTkEntry(right_frame, placeholder_text="คำศัพท์ใหม่...")
       self.new_keyword_entry.pack(pady=5)
       ctk.CTkButton(right_frame, text="เพิ่มคำศัพท์", command=self.add_single_keyword).pack(pady=5)
       # Add multiple keywords
       ctk.CTkLabel(right_frame, text="เพิ่มหลายคำพร้อมกัน:", font=ctk.CTkFont(size=12)).pack(pady=(20,2))
       self.multiple_keywords_textbox = ctk.CTkTextbox(right_frame, height=100, width=250)
       self.multiple_keywords_textbox.pack(pady=5)
       ctk.CTkButton(right_frame, text="เพิ่มหลายคำ", command=self.add_multiple_keywords).pack(pady=5)
       # Management buttons
       button_frame = ctk.CTkFrame(right_frame)
       button_frame.pack(fill="x", pady=10)
       ctk.CTkButton(button_frame, text="ลบคำที่เลือก", command=self.remove_selected_keywords, 
                    fg_color="red", hover_color="darkred").pack(pady=2)
       ctk.CTkButton(button_frame, text="ลบทั้งหมด", command=self.clear_all_keywords, 
                    fg_color="darkred", hover_color="red").pack(pady=2)
       ctk.CTkButton(button_frame, text="บันทึกคำศัพท์", command=self.save_keywords).pack(pady=2)
       # Instructions
       instructions_frame = ctk.CTkFrame(right_frame)
       instructions_frame.pack(fill="x", pady=10)
       instructions_text = """วิธีใช้:
1. พิมพ์คำศัพท์ในช่องซ้าย (1 คำต่อบรรทัด)
2. หรือพิมพ์/วางหลายคำในช่องขวา
3. เลือกข้อความที่ต้องการลบ
4. กด 'บันทึก' เมื่อเสร็จสิ้น"""
       ctk.CTkLabel(instructions_frame, text=instructions_text, 
                   justify="left", font=ctk.CTkFont(size=10)).pack(padx=5, pady=5)
       # SSML settings
       ssml_frame = ctk.CTkFrame(right_frame)
       ssml_frame.pack(fill="x", pady=10)
       ctk.CTkLabel(ssml_frame, text="การตั้งค่า SSML:", font=ctk.CTkFont(weight="bold")).pack(pady=5)
       # Rate
       ctk.CTkLabel(ssml_frame, text="ความเร็ว (Rate):").pack()
       self.rate_var = tk.StringVar()
       self.rate_entry = ctk.CTkEntry(ssml_frame, textvariable=self.rate_var, width=100)
       self.rate_entry.pack(pady=2)
       # Pitch
       ctk.CTkLabel(ssml_frame, text="ระดับเสียง (Pitch):").pack()
       self.pitch_var = tk.StringVar()
       self.pitch_entry = ctk.CTkEntry(ssml_frame, textvariable=self.pitch_var, width=100)
       self.pitch_entry.pack(pady=2)
       # Volume
       ctk.CTkLabel(ssml_frame, text="ความดัง (Volume):").pack()
       self.volume_var = tk.StringVar()
       self.volume_entry = ctk.CTkEntry(ssml_frame, textvariable=self.volume_var, width=100)
       self.volume_entry.pack(pady=2)
       ctk.CTkButton(ssml_frame, text="บันทึกการตั้งค่า", command=self.save_ssml_settings).pack(pady=5)
       # Bottom buttons
       bottom_frame = ctk.CTkFrame(main_frame)
       bottom_frame.pack(fill="x", padx=10, pady=10)
       ctk.CTkButton(bottom_frame, text="บันทึกและปิด", command=self.save_and_close).pack(side="right", padx=5)
       ctk.CTkButton(bottom_frame, text="ยกเลิก", command=self.window.destroy).pack(side="right", padx=5)
       # Load initial data
       self.load_emotion_keywords("neutral")
   def load_emotion_keywords(self, emotion):
       """โหลดคำศัพท์ของอารมณ์ที่เลือก"""
       try:
           self.keywords_textbox.delete("1.0", tk.END)
           keywords = self.emotion_analyzer.get_emotion_keywords(emotion)
           keywords_text = "\n".join(keywords)
           self.keywords_textbox.insert("1.0", keywords_text)
           # โหลด SSML settings
           ssml_config = self.emotion_analyzer.emotion_config.get(emotion, {}).get('ssml', {})
           self.rate_var.set(ssml_config.get('rate', '1.0'))
           self.pitch_var.set(ssml_config.get('pitch', '0st'))
           self.volume_var.set(ssml_config.get('volume', 'medium'))
       except Exception as e:
           print(f"Error loading emotion keywords: {e}")
   def add_single_keyword(self):
       """เพิ่มคำศัพท์เดียว"""
       try:
           keyword = self.new_keyword_entry.get().strip()
           if keyword:
               emotion = self.emotion_var.get()
               if self.emotion_analyzer.add_emotion_keyword(emotion, keyword):
                   # อัพเดท textbox
                   current_text = self.keywords_textbox.get("1.0", tk.END).strip()
                   if current_text:
                       new_text = current_text + "\n" + keyword
                   else:
                       new_text = keyword
                   self.keywords_textbox.delete("1.0", tk.END)
                   self.keywords_textbox.insert("1.0", new_text)
                   self.new_keyword_entry.delete(0, tk.END)
                   messagebox.showinfo("สำเร็จ", f"เพิ่มคำศัพท์ '{keyword}' ให้อารมณ์ '{emotion}' แล้ว")
               else:
                   messagebox.showerror("ผิดพลาด", "ไม่สามารถเพิ่มคำศัพท์ได้")
           else:
               messagebox.showwarning("คำเตือน", "กรุณาพิมพ์คำศัพท์ที่ต้องการเพิ่ม")
       except Exception as e:
           messagebox.showerror("ผิดพลาด", f"เกิดข้อผิดพลาด: {e}")
   def add_multiple_keywords(self):
       """เพิ่มหลายคำพร้อมกัน"""
       try:
           multiple_text = self.multiple_keywords_textbox.get("1.0", tk.END).strip()
           if multiple_text:
               emotion = self.emotion_var.get()
               keywords = [line.strip() for line in multiple_text.split('\n') if line.strip()]
               added_count = 0
               for keyword in keywords:
                   if self.emotion_analyzer.add_emotion_keyword(emotion, keyword):
                       added_count += 1
               if added_count > 0:
                   # อัพเดท textbox
                   current_text = self.keywords_textbox.get("1.0", tk.END).strip()
                   if current_text:
                       new_text = current_text + "\n" + "\n".join(keywords)
                   else:
                       new_text = "\n".join(keywords)
                   self.keywords_textbox.delete("1.0", tk.END)
                   self.keywords_textbox.insert("1.0", new_text)
                   self.multiple_keywords_textbox.delete("1.0", tk.END)
                   messagebox.showinfo("สำเร็จ", f"เพิ่มคำศัพท์ {added_count} คำให้อารมณ์ '{emotion}' แล้ว")
               else:
                   messagebox.showerror("ผิดพลาด", "ไม่สามารถเพิ่มคำศัพท์ได้")
           else:
               messagebox.showwarning("คำเตือน", "กรุณาพิมพ์คำศัพท์ที่ต้องการเพิ่ม")
       except Exception as e:
           messagebox.showerror("ผิดพลาด", f"เกิดข้อผิดพลาด: {e}")
   def remove_selected_keywords(self):
       """ลบคำศัพท์ที่เลือก"""
       try:
           selected_text = self.keywords_textbox.selection_get()
           if selected_text:
               emotion = self.emotion_var.get()
               keywords_to_remove = [line.strip() for line in selected_text.split('\n') if line.strip()]
               result = messagebox.askyesno("ยืนยัน", f"ต้องการลบคำศัพท์ {len(keywords_to_remove)} คำ จากอารมณ์ '{emotion}' หรือไม่?")
               if result:
                   removed_count = 0
                   for keyword in keywords_to_remove:
                       if self.emotion_analyzer.remove_emotion_keyword(emotion, keyword):
                           removed_count += 1
                   if removed_count > 0:
                       # อัพเดท textbox
                       self.load_emotion_keywords(emotion)
                       messagebox.showinfo("สำเร็จ", f"ลบคำศัพท์ {removed_count} คำแล้ว")
                   else:
                       messagebox.showerror("ผิดพลาด", "ไม่สามารถลบคำศัพท์ได้")
           else:
               messagebox.showwarning("คำเตือน", "กรุณาเลือกข้อความที่ต้องการลบ")
       except tk.TclError:
           messagebox.showwarning("คำเตือน", "กรุณาเลือกข้อความที่ต้องการลบ")
       except Exception as e:
           messagebox.showerror("ผิดพลาด", f"เกิดข้อผิดพลาด: {e}")
   def clear_all_keywords(self):
       """ลบคำศัพท์ทั้งหมด"""
       try:
           emotion = self.emotion_var.get()
           result = messagebox.askyesno("ยืนยัน", f"ต้องการลบคำศัพท์ทั้งหมดจากอารมณ์ '{emotion}' หรือไม่?")
           if result:
               keywords = self.emotion_analyzer.get_emotion_keywords(emotion)
               for keyword in keywords:
                   self.emotion_analyzer.remove_emotion_keyword(emotion, keyword)
               self.keywords_textbox.delete("1.0", tk.END)
               messagebox.showinfo("สำเร็จ", f"ลบคำศัพท์ทั้งหมดจากอารมณ์ '{emotion}' แล้ว")
       except Exception as e:
           messagebox.showerror("ผิดพลาด", f"เกิดข้อผิดพลาด: {e}")
   def save_keywords(self):
       """บันทึกคำศัพท์จาก textbox"""
       try:
           emotion = self.emotion_var.get()
           text_content = self.keywords_textbox.get("1.0", tk.END).strip()
           if text_content:
               new_keywords = [line.strip() for line in text_content.split('\n') if line.strip()]
               # ลบคำศัพท์เก่าทั้งหมด
               old_keywords = self.emotion_analyzer.get_emotion_keywords(emotion)
               for keyword in old_keywords:
                   self.emotion_analyzer.remove_emotion_keyword(emotion, keyword)
               # เพิ่มคำศัพท์ใหม่
               added_count = 0
               for keyword in new_keywords:
                   if self.emotion_analyzer.add_emotion_keyword(emotion, keyword):
                       added_count += 1
               if added_count > 0:
                   messagebox.showinfo("สำเร็จ", f"บันทึกคำศัพท์ {added_count} คำสำหรับอารมณ์ '{emotion}' แล้ว")
               else:
                   messagebox.showwarning("คำเตือน", "ไม่มีคำศัพท์ที่ถูกต้องให้บันทึก")
           else:
               # ถ้าไม่มีข้อความ แสดงว่าต้องการลบทั้งหมด
               self.clear_all_keywords()
       except Exception as e:
           messagebox.showerror("ผิดพลาด", f"เกิดข้อผิดพลาด: {e}")
   def save_ssml_settings(self):
       """บันทึกการตั้งค่า SSML"""
       try:
           emotion = self.emotion_var.get()
           rate = self.rate_var.get().strip()
           pitch = self.pitch_var.get().strip()
           volume = self.volume_var.get().strip()
           if emotion in self.emotion_analyzer.emotion_config:
               self.emotion_analyzer.emotion_config[emotion]['ssml'] = {
                   'rate': rate if rate else '1.0',
                   'pitch': pitch if pitch else '0st',
                   'volume': volume if volume else 'medium'
               }
               if self.emotion_analyzer.save_emotion_config():
                   messagebox.showinfo("สำเร็จ", f"บันทึกการตั้งค่า SSML สำหรับ '{emotion}' แล้ว")
               else:
                   messagebox.showerror("ผิดพลาด", "ไม่สามารถบันทึกการตั้งค่าได้")
           else:
               messagebox.showerror("ผิดพลาด", f"ไม่พบอารมณ์ '{emotion}' ในระบบ")
       except Exception as e:
           messagebox.showerror("ผิดพลาด", f"เกิดข้อผิดพลาด: {e}")
   def test_emotion(self):
       """ทดสอบการวิเคราะห์อารมณ์"""
       try:
           test_text = self.test_entry.get().strip()
           if not test_text:
               messagebox.showwarning("แจ้งเตือน", "กรุณาพิมพ์ข้อความเพื่อทดสอบ")
               return
           # สร้าง SSML Generator
           ssml_gen = SSMLGenerator(self.emotion_analyzer)
           # ทดสอบทั้งแบบ simple และ advanced
           simple_ssml, simple_analysis = ssml_gen.test_ssml_generation(test_text, 'simple')
           advanced_ssml, advanced_analysis = ssml_gen.test_ssml_generation(test_text, 'advanced')
           # แสดงผลลัพธ์
           result_window = ctk.CTkToplevel(self.window)
           result_window.title("ผลการทดสอบ")
           result_window.geometry("700x600")
           result_window.transient(self.window)
           # Simple analysis result
           simple_frame = ctk.CTkFrame(result_window)
           simple_frame.pack(fill="x", padx=10, pady=5)
           ctk.CTkLabel(simple_frame, text="แบบง่าย (Simple):", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
           ctk.CTkLabel(simple_frame, text=f"อารมณ์: {simple_analysis['emotion']}", anchor="w").pack(anchor="w", padx=5)
           ctk.CTkLabel(simple_frame, text=f"คะแนน: {simple_analysis['confidence']}", anchor="w").pack(anchor="w", padx=5)
           simple_text = ctk.CTkTextbox(simple_frame, height=80)
           simple_text.pack(fill="x", padx=5, pady=5)
           simple_text.insert("1.0", simple_ssml)
           simple_text.configure(state="disabled")
           # Advanced analysis result
           advanced_frame = ctk.CTkFrame(result_window)
           advanced_frame.pack(fill="both", expand=True, padx=10, pady=5)
           ctk.CTkLabel(advanced_frame, text="แบบขั้นสูง (Advanced):", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
           if advanced_analysis.get('mode') == 'advanced':
               for i, sentence_result in enumerate(advanced_analysis['sentences']):
                   ctk.CTkLabel(advanced_frame, 
                              text=f"ประโยคที่ {i+1}: {sentence_result['emotion']} (คะแนน: {sentence_result['confidence']})", 
                              anchor="w").pack(anchor="w", padx=5)
           advanced_text = ctk.CTkTextbox(advanced_frame, height=120)
           advanced_text.pack(fill="both", expand=True, padx=5, pady=5)
           advanced_text.insert("1.0", advanced_ssml)
           advanced_text.configure(state="disabled")
       except Exception as e:
           messagebox.showerror("ผิดพลาด", f"เกิดข้อผิดพลาดในการทดสอบ: {e}")
   def save_and_close(self):
       """บันทึกและปิดหน้าต่าง"""
       try:
           # บันทึกคำศัพท์จาก textbox ก่อน
           self.save_keywords()
           if self.emotion_analyzer.save_emotion_config():
               messagebox.showinfo("สำเร็จ", "บันทึกการตั้งค่าเรียบร้อยแล้ว")
               self.window.destroy()
           else:
               messagebox.showerror("ผิดพลาด", "ไม่สามารถบันทึกการตั้งค่าได้")
       except Exception as e:
           messagebox.showerror("ผิดพลาด", f"เกิดข้อผิดพลาด: {e}")
# ===============================================
# TEXT PROCESSING UTILITIES
# ===============================================
class RegexMode(Enum):
  """Enum สำหรับโหมดการตัดประโยคด้วย Regex"""
  WHITESPACE = auto()
  SIMPLE = auto()
  ADVANCED = auto()
  THAI_OPTIMIZED = auto()
  CUSTOM = auto()
# Dictionary สำหรับเก็บ Regex Pattern ของแต่ละโหมด
REGEX_PATTERNS = {
  RegexMode.WHITESPACE: r'\s+',
  RegexMode.SIMPLE: r'[.!?।|]+',
  RegexMode.ADVANCED: r'[.!?।|]+(?=\s|\n|$)',
  RegexMode.THAI_OPTIMIZED: r'[.!?।|]+|(?<=\s)(?=\S)',
}
# Dictionary สำหรับเก็บคำอธิบายของแต่ละโหมด
REGEX_DESCRIPTIONS = {
  RegexMode.WHITESPACE: "แบ่งทุกครั้งที่เจอช่องว่าง (Whitespace) เหมาะสำหรับข้อความที่ไม่มีเครื่องหมายวรรคตอนเลย (การทำงานแบบเดิม)",
  RegexMode.SIMPLE: "แบ่งเมื่อเจอเครื่องหมาย . ! ? । หรือ | เหมาะสำหรับประโยคภาษาอังกฤษทั่วไป",
  RegexMode.ADVANCED: "เหมือนแบบ Simple แต่จะฉลาดกว่า โดยจะแบ่งเมื่อเจอเครื่องหมายวรรคตอนที่ตามด้วยช่องว่างหรือขึ้นบรรทัดใหม่",
  RegexMode.THAI_OPTIMIZED: "ออกแบบมาสำหรับภาษาไทยโดยเฉพาะ จะแบ่งตามเครื่องหมายวรรคตอน หรือแบ่งระหว่างคำถ้าไม่มีเครื่องหมาย",
  RegexMode.CUSTOM: "กำหนดรูปแบบ Regex เอง (สำหรับผู้ใช้ขั้นสูง) โปรดระวังการใช้รูปแบบที่ผิดพลาด",
}
def calculate_text_bytes(text):
  """คำนวณขนาด byte ของข้อความ (UTF-8)"""
  return len(text.encode('utf-8'))
def split_text_by_sentences(text, mode, custom_pattern=None, max_bytes=900):
  """แบ่งข้อความตามประโยค โดยไม่เกิน max_bytes ต่อส่วน และรองรับ Regex หลายโหมด"""
  sentence_endings = REGEX_PATTERNS.get(mode)
  if mode == RegexMode.CUSTOM:
      sentence_endings = custom_pattern
  if not sentence_endings:
      # Fallback to default if mode is invalid
      sentence_endings = REGEX_PATTERNS[RegexMode.WHITESPACE]
  try:
      # แยกประโยคด้วย sentence endings ที่เลือก
      sentences = re.split(f'({sentence_endings})', text)
  except re.error as e:
      # หาก Custom Regex ผิดพลาด ให้กลับไปใช้ค่าเริ่มต้นและแจ้งเตือน
      print(f"Custom Regex Error: {e}. Falling back to WHITESPACE mode.")
      sentences = re.split(f'({REGEX_PATTERNS[RegexMode.WHITESPACE]})', text)
  # รวมประโยคกับเครื่องหมายจบประโยค
  complete_sentences = []
  for i in range(0, len(sentences) - 1, 2):
      if i + 1 < len(sentences):
          complete_sentences.append(sentences[i] + sentences[i + 1])
      else:
          complete_sentences.append(sentences[i])
  # จัดกลุ่มประโยคไม่ให้เกิน max_bytes (โค้ดส่วนนี้เหมือนเดิม)
  chunks = []
  current_chunk = ""
  for sentence in complete_sentences:
      sentence = sentence.strip()
      if not sentence:
          continue
      test_chunk = current_chunk + " " + sentence if current_chunk else sentence
      if calculate_text_bytes(test_chunk) <= max_bytes:
          current_chunk = test_chunk
      else:
          if calculate_text_bytes(sentence) > max_bytes:
              if current_chunk:
                  chunks.append(current_chunk.strip())
                  current_chunk = ""
              chunks.extend(split_long_sentence(sentence, max_bytes))
          else:
              if current_chunk:
                  chunks.append(current_chunk.strip())
              current_chunk = sentence
  if current_chunk:
      chunks.append(current_chunk.strip())
  return chunks
def split_long_sentence(sentence, max_bytes=900):
  """แบ่งประโยคยาวๆ ตาม word boundaries"""
  words = sentence.split()
  chunks = []
  current_chunk = ""
  for word in words:
      test_chunk = current_chunk + " " + word if current_chunk else word
      if calculate_text_bytes(test_chunk) <= max_bytes:
          current_chunk = test_chunk
      else:
          if current_chunk:
              chunks.append(current_chunk.strip())
          current_chunk = word
          # หากคำเดียวเกิน max_bytes (กรณีพิเศษ)
          if calculate_text_bytes(word) > max_bytes:
              # แบ่งตัวอักษร (กรณีสุดท้าย)
              for i in range(0, len(word), 100):
                  chunks.append(word[i:i+100])
              current_chunk = ""
  if current_chunk:
      chunks.append(current_chunk.strip())
  return chunks
def merge_audio_files(audio_file_paths, output_path):
  """รวมไฟล์เสียงหลายไฟล์เป็นไฟล์เดียว"""
  try:
      combined = AudioSegment.empty()
      for audio_path in audio_file_paths:
          if os.path.exists(audio_path):
              audio_segment = AudioSegment.from_file(audio_path)
              combined += audio_segment
              # เพิ่มช่วงเงียบเล็กน้อยระหว่างส่วน (0.2 วินาที)
              combined += AudioSegment.silent(duration=200)
      # Export ไฟล์รวม
      combined.export(output_path, format=os.path.splitext(output_path)[1][1:])
      # ลบไฟล์ชั่วคราว
      for temp_file in audio_file_paths:
          try:
              os.remove(temp_file)
          except:
              pass
      return True
  except Exception as e:
      print(f"Error merging audio files: {e}")
      return False
# ===============================================
# ENHANCED EMOTION CONTROL FOR TTS
# ===============================================
def create_emotion_ssml(text, emotion="neutral"):
  """สร้าง SSML สำหรับอารมณ์ต่าง ๆ"""
  emotion_configs = {
      "sad": {
          "rate": "0.75",
          "pitch": "-3st", 
          "volume": "soft"
      },
      "excited": {
          "rate": "1.2",
          "pitch": "+2st",
          "volume": "loud"
      },
      "angry": {
          "rate": "1.1", 
          "pitch": "+3st",
          "volume": "loud"
      },
      "calm": {
          "rate": "0.85",
          "pitch": "-1st",
          "volume": "medium"
      },
      "whisper": {
          "rate": "0.6",
          "pitch": "-2st", 
          "volume": "x-soft"
      },
      "neutral": {
          "rate": "1.0",
          "pitch": "0st",
          "volume": "medium"
      }
  }
  if emotion in emotion_configs:
      config = emotion_configs[emotion]
      ssml = f'''<speak>
          <prosody rate="{config['rate']}" 
                   pitch="{config['pitch']}" 
                   volume="{config['volume']}">
              {text}
          </prosody>
      </speak>'''
  else:
      ssml = f"<speak>{text}</speak>"
  return ssml
def create_auto_emotion_ssml(text, emotion_analyzer, ssml_generator, use_advanced=False):
   """สร้าง SSML อัตโนมัติจากการวิเคราะห์อารมณ์ (ใช้คำแรกที่พบ)"""
   try:
       if use_advanced:
           analysis = emotion_analyzer.analyze_advanced(text)
           ssml = ssml_generator.create_advanced_ssml(analysis)
       else:
           analysis = emotion_analyzer.analyze_simple(text)
           ssml = ssml_generator.create_simple_ssml(analysis)
       # บันทึก log การวิเคราะห์
       log_emotion_analysis(text, analysis, ssml)
       return ssml, analysis
   except Exception as e:
       print(f"Error in auto emotion SSML generation: {e}")
       # ถ้าเกิดข้อผิดพลาด ให้ใช้ข้อความธรรมดา
       return f"<speak>{text}</speak>", {"emotion": "neutral", "error": str(e)}
def log_emotion_analysis(text, analysis, ssml, log_file=EMOTION_LOG_FILE):
   """บันทึก log การวิเคราะห์อารมณ์แบบละเอียด"""
   try:
       timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
       # สร้างข้อมูล log ที่ละเอียดขึ้น
       log_entry = {
           "timestamp": timestamp,
           "original_text": text,
           "analysis": analysis,
           "ssml": ssml,
           "analysis_summary": {
               "mode": analysis.get('mode', 'simple'),
               "primary_emotion": analysis.get('emotion', 'neutral'),
               "found_keywords": analysis.get('analysis_info', {}).get('found_keywords', []),
               "emotion_distribution": analysis.get('analysis_info', {}).get('emotion_distribution', {})
           }
       }
       with open(log_file, "a", encoding="utf-8") as f:
           f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
   except Exception as e:
       print(f"Error logging emotion analysis: {e}")
# ===============================================
# GOOGLE CLOUD TTS QUOTA MANAGEMENT SYSTEM
# ===============================================
# Google TTS Quota limits และการตั้งค่า
# Google TTS Quota limits และการตั้งค่า
GOOGLE_TTS_QUOTA_LIMITS = {
    'standard': {'limit': 3900000, 'unit': 'characters', 'name': 'Standard Voice'},
    'wavenet': {'limit': 900000, 'unit': 'characters', 'name': 'WaveNet Voice'},
    'neural2': {'limit': 900000, 'unit': 'bytes', 'name': 'Neural2 Voice'},
    'chirp': {'limit': 900000, 'unit': 'characters', 'name': 'Chirp Voice'},
    'hd': {'limit': 900000, 'unit': 'characters', 'name': 'HD Voice'},
    'studio': {'limit': 90000, 'unit': 'bytes', 'name': 'Studio Voice'}
}
GOOGLE_TTS_QUOTA_FILE = "google_tts_quota.json"
class GoogleTTSQuotaManager:
    """
    คลาสจัดการ quota สำหรับ Google Cloud Text-to-Speech (เวอร์ชันปรับปรุงสำหรับหลาย API keys)
    โครงสร้างข้อมูลใหม่ใน google_tts_quota.json:
    {
        "C:\\path\\to\\key1.json": {
            "month": 7,
            "year": 2025,
            "last_updated": "...",
            "paid_features": { "neural2": false, ... },
            "usage": {
                "neural2": 15000,
                "standard": 20000
            }
        },
        "C:\\path\\to\\key2.json": { ... }
    }
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.quota_file = os.path.join(output_dir, GOOGLE_TTS_QUOTA_FILE)
        self.all_quota_data = self.load_quota_data()
    def _get_default_key_structure(self):
        """สร้างโครงสร้างข้อมูลเริ่มต้นสำหรับ key ใหม่"""
        now = datetime.datetime.now()
        return {
            'month': now.month,
            'year': now.year,
            'last_updated': now.isoformat(),
            'paid_features': {
                'standard': False, 'wavenet': False, 'neural2': False,
                'chirp': False, 'hd': False, 'studio': False
            },
            'usage': {}
        }
    def _get_key_data(self, key_path, create_if_missing=False):
        """ดึงข้อมูลของ key ที่ระบุ หรือสร้างใหม่ถ้ายังไม่มี"""
        if key_path not in self.all_quota_data:
            if create_if_missing:
                self.all_quota_data[key_path] = self._get_default_key_structure()
            else:
                return None
        # ตรวจสอบและรีเซ็ตโควต้ารายเดือนสำหรับ key นี้
        key_data = self.all_quota_data[key_path]
        now = datetime.datetime.now()
        if key_data.get('month') != now.month or key_data.get('year') != now.year:
            print(f"New month detected for key {os.path.basename(key_path)}. Resetting quota.")
            key_data['usage'] = {}
            key_data['month'] = now.month
            key_data['year'] = now.year
            key_data['reset_date'] = now.isoformat()
        return key_data
    def get_voice_type(self, voice_name):
        """ตรวจสอบประเภทเสียงจากชื่อ"""
        if not voice_name:
            return 'standard'
        voice_name_lower = voice_name.lower()
        if 'studio' in voice_name_lower:
            return 'studio'
        elif 'neural2' in voice_name_lower:
            return 'neural2'
        elif 'wavenet' in voice_name_lower:
            return 'wavenet'
        elif 'chirp' in voice_name_lower:
            return 'chirp'
        elif 'hd' in voice_name_lower or 'journey' in voice_name_lower:
            return 'hd'
        else:
            return 'standard'
    def calculate_usage(self, text, voice_type):
        """คำนวณการใช้งาน (characters หรือ bytes)"""
        if not text:
            return 0
        if voice_type in ['neural2', 'studio']:
            return calculate_text_bytes(text)
        else:
            return len(text)
    def check_quota(self, text, voice_name, key_path):
        """ตรวจสอบ quota ก่อนสร้างเสียงสำหรับ key ที่ระบุ"""
        if not key_path:
            return False, {'error': "No key path provided"}
        key_data = self._get_key_data(key_path, create_if_missing=True)
        voice_type = self.get_voice_type(voice_name)
        usage = self.calculate_usage(text, voice_type)
        current_usage = key_data['usage'].get(voice_type, 0)
        limit_info = GOOGLE_TTS_QUOTA_LIMITS.get(voice_type, GOOGLE_TTS_QUOTA_LIMITS['standard'])
        limit = limit_info['limit']
        if current_usage + usage > limit:
            if not key_data['paid_features'].get(voice_type, False):
                return False, {
                    'key_path': key_path,
                    'voice_type': voice_type,
                    'current_usage': current_usage,
                    'requested_usage': usage,
                    'total_would_be': current_usage + usage,
                    'limit': limit,
                    'unit': limit_info['unit'],
                    'over_limit': (current_usage + usage) - limit
                }
        return True, {
            'key_path': key_path,
            'voice_type': voice_type,
            'current_usage': current_usage,
            'requested_usage': usage,
            'total_would_be': current_usage + usage,
            'limit': limit,
            'unit': limit_info['unit']
        }
    def update_usage(self, text, voice_name, key_path):
        """อัพเดทการใช้งานสำหรับ key ที่ระบุ"""
        if not key_path:
            return {}
        key_data = self._get_key_data(key_path, create_if_missing=True)
        voice_type = self.get_voice_type(voice_name)
        usage = self.calculate_usage(text, voice_type)
        if voice_type not in key_data['usage']:
            key_data['usage'][voice_type] = 0
        key_data['usage'][voice_type] += usage
        key_data['last_updated'] = datetime.datetime.now().isoformat()
        self.save_quota_data()
        return {
            'key_path': key_path,
            'voice_type': voice_type,
            'usage_added': usage,
            'total_usage': key_data['usage'][voice_type],
            'unit': GOOGLE_TTS_QUOTA_LIMITS.get(voice_type, {}).get('unit', 'characters')
        }
    def get_full_usage_summary(self):
        """ได้รับสรุปการใช้งานทั้งหมด โดยแยกตาม API key"""
        summary_lines = []
        if not self.all_quota_data:
            return "No quota data available."
        for key_path, key_data in self.all_quota_data.items():
            key_name = os.path.basename(key_path)
            summary_lines.append(f"===== Quota Summary for Key: {key_name} =====")
            usage_found = False
            for voice_type, limit_info in GOOGLE_TTS_QUOTA_LIMITS.items():
                current_usage = key_data.get('usage', {}).get(voice_type, 0)
                if current_usage > 0:
                    usage_found = True
                    limit = limit_info['limit']
                    unit = limit_info['unit']
                    percentage = (current_usage / limit) * 100 if limit > 0 else 0
                    paid_enabled = key_data.get('paid_features', {}).get(voice_type, False)
                    summary_lines.append(
                        f"  - {limit_info['name']} ({voice_type}): "
                        f"{current_usage:,}/{limit:,} {unit} ({percentage:.1f}%)"
                        f" [Paid: {'ON' if paid_enabled else 'OFF'}]"
                    )
            if not usage_found:
                summary_lines.append("  (No usage recorded for this key in the current month)")
            summary_lines.append("") # Add a blank line for readability
        return "\n".join(summary_lines)
    def enable_paid_feature(self, voice_type, key_path, enabled=True):
        """เปิด/ปิด paid feature สำหรับประเภทเสียงของ key ที่ระบุ"""
        if not key_path:
            return
        key_data = self._get_key_data(key_path, create_if_missing=True)
        if voice_type in key_data['paid_features']:
            key_data['paid_features'][voice_type] = enabled
            self.save_quota_data()
        else:
            print(f"Warning: Unknown voice type '{voice_type}' for paid feature")
    def save_quota_data(self):
        """บันทึกข้อมูล quota ทั้งหมด"""
        try:
            with open(self.quota_file, 'w', encoding='utf-8') as f:
                json.dump(self.all_quota_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving quota data: {e}")
            return False
    def load_quota_data(self):
        """โหลดข้อมูล quota ทั้งหมด"""
        if not os.path.exists(self.quota_file):
            return {}
        try:
            with open(self.quota_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            # ตรวจสอบและรีเซ็ตโควต้าสำหรับแต่ละ key
            now = datetime.datetime.now()
            for key_path, key_data in loaded_data.items():
                if key_data.get('month') != now.month or key_data.get('year') != now.year:
                    print(f"New month detected for key {os.path.basename(key_path)}. Resetting quota.")
                    key_data['usage'] = {}
                    key_data['month'] = now.month
                    key_data['year'] = now.year
                    key_data['reset_date'] = now.isoformat()
            return loaded_data
        except Exception as e:
            print(f"Error loading quota data: {e}, creating fresh data.")
            return {}
    def get_realtime_display_for_key(self, voice_type, key_path):
        """ได้รับข้อมูลสำหรับแสดงผล realtime สำหรับ key ที่ระบุ"""
        if not key_path:
            return {
                'formatted': "No Key Selected", 'percentage': 0, 'paid_enabled': False
            }
        key_data = self._get_key_data(key_path, create_if_missing=True)
        limit_info = GOOGLE_TTS_QUOTA_LIMITS.get(voice_type, {})
        current_usage = key_data.get('usage', {}).get(voice_type, 0)
        limit = limit_info.get('limit', 0)
        unit = limit_info.get('unit', 'characters')
        percentage = (current_usage / limit) * 100 if limit > 0 else 0
        name = limit_info.get('name', voice_type)
        paid_enabled = key_data.get('paid_features', {}).get(voice_type, False)
        if unit == 'characters':
            current_display = f"{current_usage:,}"
            limit_display = f"{limit:,}"
        else:  # bytes
            current_display = f"{current_usage:,}"
            limit_display = f"{limit:,}"
        formatted = f"{name}: {current_display}/{limit_display} {unit} ({percentage:.1f}%)"
        return {
            'current_usage': current_usage,
            'limit': limit,
            'unit': unit,
            'percentage': percentage,
            'remaining': max(0, limit - current_usage),
            'paid_enabled': paid_enabled,
            'formatted': formatted
        }
# Standalone utility functions (ปรับปรุงให้รับ key_path)
def create_quota_manager(output_dir):
    """สร้าง quota manager instance"""
    if not output_dir or not os.path.exists(output_dir):
        return None
    try:
        return GoogleTTSQuotaManager(output_dir)
    except Exception as e:
        print(f"Error creating quota manager: {e}")
        return None
def check_google_tts_quota(text, voice_name, output_dir, key_path):
    """ตรวจสอบ quota สำหรับการสร้างเสียง (ต้องการ key_path)"""
    quota_manager = create_quota_manager(output_dir)
    if not quota_manager:
        return True, {}
    return quota_manager.check_quota(text, voice_name, key_path)
def update_google_tts_usage(text, voice_name, output_dir, key_path):
    """อัพเดทการใช้งาน Google TTS (ต้องการ key_path)"""
    quota_manager = create_quota_manager(output_dir)
    if not quota_manager:
        return {}
    return quota_manager.update_usage(text, voice_name, key_path)
def get_google_tts_usage_summary(output_dir):
    """ได้รับสรุปการใช้งาน Google TTS ทั้งหมด"""
    quota_manager = create_quota_manager(output_dir)
    if not quota_manager:
        return "Quota manager could not be created."
    return quota_manager.get_full_usage_summary()
def handle_quota_exceeded_error(quota_info, voice_name, log_widget=None):
    """จัดการข้อผิดพลาดเมื่อเกิน quota"""
    key_path = quota_info.get('key_path', 'Unknown Key')
    voice_type = quota_info.get('voice_type', 'unknown')
    current = quota_info.get('current_usage', 0)
    requested = quota_info.get('requested_usage', 0)
    limit = quota_info.get('limit', 0)
    unit = quota_info.get('unit', 'characters')
    over_limit = quota_info.get('over_limit', 0)
    error_msg = f"Google TTS Quota exceeded for key: {os.path.basename(key_path)}\n\n"
    error_msg += f"Voice Type: {voice_type}\n"
    error_msg += f"Current usage: {current:,} {unit}\n"
    error_msg += f"Requested: {requested:,} {unit}\n"
    error_msg += f"Monthly limit: {limit:,} {unit}\n"
    error_msg += f"Would exceed by: {over_limit:,} {unit}\n\n"
    error_msg += f"Please enable paid feature for this key or wait for next month."
    if log_widget:
        log_message(log_widget, f"Quota exceeded for key {os.path.basename(key_path)} ({voice_name}): {current:,}/{limit:,} {unit}", level="ERROR")
    return error_msg
def show_quota_warning(quota_info, voice_name):
    """แสดงคำเตือนเกี่ยวกับ quota"""
    key_path = quota_info.get('key_path', 'Unknown Key')
    voice_type = quota_info.get('voice_type', 'unknown')
    current = quota_info.get('current_usage', 0)
    limit = quota_info.get('limit', 0)
    percentage = (current / limit) * 100 if limit > 0 else 0
    if percentage >= 90:
        warning_msg = f"⚠️ Google TTS Quota Warning for key '{os.path.basename(key_path)}' ({voice_type} voice):\n"
        warning_msg += f"Usage: {current:,}/{limit:,} {quota_info.get('unit', 'characters')} ({percentage:.1f}%)\n"
        warning_msg += f"Voice: {voice_name}"
        return warning_msg
    return None
# ===============================================
# SYSTEM UTILITIES
# ===============================================
def shutdown_windows():
   """Shutdown Windows system"""
   try:
       if os.name == 'nt':  # Windows
           subprocess.run(['shutdown', '/s', '/t', '30'], check=True)
           return True
   except Exception as e:
       print(f"Failed to shutdown Windows: {e}")
       return False
# ===============================================
# PROGRESS TRACKING UTILITIES
# ===============================================
def create_progress_tracking_dir(output_dir):
  """Create progress tracking directory"""
  progress_dir = os.path.join(output_dir, PROGRESS_DIR)
  os.makedirs(progress_dir, exist_ok=True)
  return progress_dir
def save_progress(output_dir, progress_file, data):
  """Save progress data to file"""
  try:
      progress_dir = create_progress_tracking_dir(output_dir)
      progress_path = os.path.join(progress_dir, progress_file)
      with open(progress_path, 'w', encoding='utf-8') as f:
          json.dump(data, f, ensure_ascii=False, indent=2)
      return True
  except Exception as e:
      print(f"Failed to save progress: {e}")
      return False
def load_progress(output_dir, progress_file):
  """Load progress data from file"""
  try:
      progress_dir = os.path.join(output_dir, PROGRESS_DIR)
      progress_path = os.path.join(progress_dir, progress_file)
      if os.path.exists(progress_path):
          with open(progress_path, 'r', encoding='utf-8') as f:
              return json.load(f)
      return None
  except Exception as e:
      print(f"Failed to load progress: {e}")
      return None
def clear_progress(output_dir, progress_file):
  """Clear progress file"""
  try:
      progress_dir = os.path.join(output_dir, PROGRESS_DIR)
      progress_path = os.path.join(progress_dir, progress_file)
      if os.path.exists(progress_path):
          os.remove(progress_path)
      return True
  except Exception as e:
      print(f"Failed to clear progress: {e}")
      return False
# ===============================================
# TIME TRACKING UTILITIES
# ===============================================
def load_time_tracking_data(output_dir):
   """Load time tracking data from file"""
   try:
       time_file_path = os.path.join(output_dir, TIME_TRACKING_FILE)
       if os.path.exists(time_file_path):
           with open(time_file_path, 'r', encoding='utf-8') as f:
               return json.load(f)
       return {}
   except Exception as e:
       print(f"Failed to load time tracking data: {e}")
       return {}
def save_time_tracking_data(output_dir, time_data):
   """Save time tracking data to file"""
   try:
       time_file_path = os.path.join(output_dir, TIME_TRACKING_FILE)
       with open(time_file_path, 'w', encoding='utf-8') as f:
           json.dump(time_data, f, ensure_ascii=False, indent=2)
       return True
   except Exception as e:
       print(f"Failed to save time tracking data: {e}")
       return False
def format_time_duration(seconds):
   """Format seconds into Month:Day:Hour:Minute format"""
   if seconds < 0:
       return "0 Month : 0 Day : 0 Hr : 0 Minute"
   minutes = int(seconds // 60)
   hours = minutes // 60
   days = hours // 24
   months = days // 30
   remaining_days = days % 30
   remaining_hours = hours % 24
   remaining_minutes = minutes % 60
   return f"{months} Month : {remaining_days} Day : {remaining_hours} Hr : {remaining_minutes:02d} Minute"
def start_task_timer(task_name, output_dir):
   """Start timer for a task"""
   global current_task_start_time, accumulated_task_times
   current_task_start_time = time.time()
   # Load existing time data
   time_data = load_time_tracking_data(output_dir)
   accumulated_task_times = time_data.get('accumulated_times', {})
   return current_task_start_time
def update_task_timer(task_name, output_dir, status_callback=None):
   """Update task timer and save progress"""
   global current_task_start_time, accumulated_task_times
   if current_task_start_time is None:
       return 0, 0
   current_time = time.time()
   current_session_time = current_time - current_task_start_time
   # Add to accumulated time
   if task_name not in accumulated_task_times:
       accumulated_task_times[task_name] = 0
   total_time = accumulated_task_times[task_name] + current_session_time
   # Save to file periodically (every 30 seconds)
   if int(current_session_time) % 30 == 0:
       time_data = {
           'accumulated_times': accumulated_task_times.copy(),
           'last_updated': datetime.datetime.now().isoformat()
       }
       time_data['accumulated_times'][task_name] = total_time
       save_time_tracking_data(output_dir, time_data)
   return total_time, current_session_time
def stop_task_timer(task_name, output_dir):
   """Stop timer for a task and save final time"""
   global current_task_start_time, accumulated_task_times
   if current_task_start_time is None:
       return 0
   current_time = time.time()
   session_time = current_time - current_task_start_time
   # Add to accumulated time
   if task_name not in accumulated_task_times:
       accumulated_task_times[task_name] = 0
   accumulated_task_times[task_name] += session_time
   # Save final time
   time_data = {
       'accumulated_times': accumulated_task_times.copy(),
       'last_updated': datetime.datetime.now().isoformat()
   }
   save_time_tracking_data(output_dir, time_data)
   current_task_start_time = None
   return accumulated_task_times[task_name]
def estimate_completion_time(current_progress, total_time_so_far):
   """Estimate completion time based on current progress"""
   if current_progress <= 0.001:  # ป้องกันการหารด้วย 0
       return 0
   # ถ้า progress น้อยมาก (<1%) ให้ประมาณการจากเวลาที่ผ่านมา
   if current_progress < 0.01:
       # ประมาณการคร่าวๆ จากเวลาที่ใช้ไปแล้ว
       rough_estimate = total_time_so_far * 100  # สมมติว่าใช้เวลา 100 เท่า
       return max(0, rough_estimate)
   estimated_total_time = total_time_so_far / current_progress
   remaining_time = estimated_total_time - total_time_so_far
   return max(0, remaining_time)
# ===============================================
# GENDER FREQUENCY RANGE UTILITIES
# ===============================================
def load_gender_frequency_settings():
   """Load gender frequency range settings"""
   config = configparser.ConfigParser()
   try:
       if os.path.exists(SETTINGS_FILE):
           config.read(SETTINGS_FILE, encoding='utf-8')
           if 'GenderFrequency' in config:
               male_min = config.getfloat('GenderFrequency', 'male_min_hz', fallback=DEFAULT_MALE_MIN_HZ)
               male_max = config.getfloat('GenderFrequency', 'male_max_hz', fallback=DEFAULT_MALE_MAX_HZ)
               female_min = config.getfloat('GenderFrequency', 'female_min_hz', fallback=DEFAULT_FEMALE_MIN_HZ)
               female_max = config.getfloat('GenderFrequency', 'female_max_hz', fallback=DEFAULT_FEMALE_MAX_HZ)
               return male_min, male_max, female_min, female_max
   except Exception as e:
       print(f"Error loading gender frequency settings: {e}")
   return DEFAULT_MALE_MIN_HZ, DEFAULT_MALE_MAX_HZ, DEFAULT_FEMALE_MIN_HZ, DEFAULT_FEMALE_MAX_HZ
def save_gender_frequency_settings(male_min, male_max, female_min, female_max):
   """Save gender frequency range settings"""
   config = configparser.ConfigParser()
   try:
       if os.path.exists(SETTINGS_FILE):
           config.read(SETTINGS_FILE, encoding='utf-8')
       if 'GenderFrequency' not in config:
           config.add_section('GenderFrequency')
       config.set('GenderFrequency', 'male_min_hz', str(male_min))
       config.set('GenderFrequency', 'male_max_hz', str(male_max))
       config.set('GenderFrequency', 'female_min_hz', str(female_min))
       config.set('GenderFrequency', 'female_max_hz', str(female_max))
       with open(SETTINGS_FILE, 'w', encoding='utf-8') as configfile:
           config.write(configfile)
       return True
   except Exception as e:
       print(f"Error saving gender frequency settings: {e}")
       return False
def estimate_gender_from_f0_custom(f0_mean, male_min, male_max, female_min, female_max):
   """Estimate gender using custom frequency ranges"""
   if male_min <= f0_mean <= male_max:
       return "Male"
   elif female_min <= f0_mean <= female_max:
       return "Female"
   else:
       return "Unknown"
def estimate_gender_from_f0(f0_mean, male_min=None, male_max=None, female_min=None, female_max=None):
   """Estimate gender from F0 with custom ranges"""
   if male_min is None:
       male_min, male_max, female_min, female_max = load_gender_frequency_settings()
   return estimate_gender_from_f0_custom(f0_mean, male_min, male_max, female_min, female_max)
def format_output_line(filename, gender, age, channel, text):
  gender_str = gender if gender in ["Male", "Female"] else "Unknown"
  age_str = str(age) if isinstance(age, (int, str)) and str(age).isdigit() else "0"
  channel_str = str(channel) if channel is not None else "0"
  return f"{filename}|{gender_str}|{age_str}|{channel_str}|{text}"
def parse_output_line(line):
  parts = line.strip().split('|', 4)
  if len(parts) == 5:
      filename, gender, age, channel, text = parts
      return filename, gender, age, channel, text
  elif len(parts) == 4:
      filename, gender, age, text = parts
      return filename, gender, age, "0", text
  else:
      return None, None, None, None, line.strip()
# ===============================================
# LOGGING UTILITIES
# ===============================================
def log_message(log_widget, message, also_print=True, level="INFO"):
   now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   formatted_message = f"{now} [{level}] {message}\n"
   if log_widget:
       log_widget.after(0, lambda: update_log_widget(log_widget, formatted_message))
   if also_print:
       print(formatted_message, end="")
   try:
       with open(os.path.join(output_folder_path if output_folder_path else ".", LOG_FILE), "a", encoding="utf-8") as f:
           f.write(formatted_message)
   except Exception as e:
       print(f"Error writing to log file: {e}")
def update_log_widget(log_widget, formatted_message):
   """Update log widget without causing flicker"""
   try:
       log_widget.configure(state='normal')
       log_widget.insert(tk.END, formatted_message)
       log_widget.configure(state='disabled')
       log_widget.see(tk.END)
   except Exception:
       pass
# ===============================================
# FILE DIALOG UTILITIES
# ===============================================
def select_folder(entry_widget, title="Select Folder"):
   folder_selected = filedialog.askdirectory(title=title)
   if folder_selected:
       entry_widget.delete(0, tk.END)
       entry_widget.insert(0, folder_selected)
       return folder_selected
   return None
def select_file(entry_widget, title="Select File", filetypes=(("All files", "*.*"),)):
   file_selected = filedialog.askopenfilename(title=title, filetypes=filetypes)
   if file_selected:
       entry_widget.delete(0, tk.END)
       entry_widget.insert(0, file_selected)
       return file_selected
   return None
# ===============================================
# API KEY VALIDATION
# ===============================================
def validate_google_json_key(key_path, status_label):
   if not key_path or not os.path.exists(key_path):
       status_label.configure(text="Key File Not Found", text_color="red")
       return False
   try:
       if GOOGLE_STT_AVAILABLE:
            speech.SpeechClient.from_service_account_json(key_path)
       if GOOGLE_TTS_AVAILABLE:
           texttospeech.TextToSpeechClient.from_service_account_json(key_path)
       status_label.configure(text="Key OK (Basic Check)", text_color="green")
       return True
   except Exception as e:
       status_label.configure(text=f"Key Invalid: {e}", text_color="red")
       return False
def validate_gemini_api_key(api_key, status_label):
   if not api_key:
       status_label.configure(text="No API Key", text_color="red")
       return False
   if not GEMINI_AVAILABLE:
       status_label.configure(text="Gemini Lib Not Installed", text_color="red")
       return False
   try:
       genai.configure(api_key=api_key)
       models = genai.list_models()
       if models:
           status_label.configure(text="Key Validated", text_color="green")
           return True
       else:
            status_label.configure(text="Key Valid (No models listed?)", text_color="orange")
            return True
   except Exception as e:
       status_label.configure(text=f"Key Invalid: {type(e).__name__}", text_color="red")
       return False
# ===============================================
# AUDIO ANALYSIS UTILITIES
# ===============================================
def estimate_gender_from_f0(f0_mean, male_min=None, male_max=None, female_min=None, female_max=None):
    """Estimate gender from F0 with custom ranges"""
    if male_min is None:
        male_min, male_max, female_min, female_max = load_gender_frequency_settings()
    return estimate_gender_from_f0_custom(f0_mean, male_min, male_max, female_min, female_max)
def format_output_line(filename, gender, age, channel, text):
   gender_str = gender if gender in ["Male", "Female"] else "Unknown"
   age_str = str(age) if isinstance(age, (int, str)) and str(age).isdigit() else "0"
   channel_str = str(channel) if channel is not None else "0"
   return f"{filename}|{gender_str}|{age_str}|{channel_str}|{text}"
def parse_output_line(line):
   parts = line.strip().split('|', 4)
   if len(parts) == 5:
       filename, gender, age, channel, text = parts
       return filename, gender, age, channel, text
   elif len(parts) == 4:
       filename, gender, age, text = parts
       return filename, gender, age, "0", text
   else:
       return None, None, None, None, line.strip()
# ===============================================
# GEMINI API CONFIGURATION
# ===============================================
def get_gemini_api_key_from_ini(config_path, log_widget):
   if not os.path.exists(config_path):
       log_message(log_widget, f"Gemini settings file not found: {config_path}", level="ERROR")
       return None, None, None, None, 1.2, "{audio_prompt}"
   config = configparser.ConfigParser()
   try:
       config.read(config_path, encoding='utf-8')
       api_keys = {k: v for k, v in config.items('API_KEYS')}
       if not api_keys:
           log_message(log_widget, "No API keys found in [API_KEYS] section.", level="ERROR")
           return None, None, None, None, 1.2, "{audio_prompt}"
       last_key_index = config.getint('STATE', 'last_key', fallback=0)
       current_key_index = (last_key_index % len(api_keys)) + 1
       current_key_name = f"api_key{current_key_index}"
       while current_key_name not in api_keys and current_key_index <= len(api_keys):
            current_key_index +=1
            current_key_name = f"api_key{current_key_index}"
            if current_key_index > len(api_keys):
                current_key_index = 1
                current_key_name = f"api_key{current_key_index}"
                if current_key_name not in api_keys:
                     log_message(log_widget, "Could not find a valid key index after checking all.", level="ERROR")
                     return None, None, None, None, 1.2, "{audio_prompt}"
       selected_key = api_keys.get(current_key_name)
       config.set('STATE', 'last_key', str(current_key_index))
       with open(config_path, 'w', encoding='utf-8') as configfile:
           config.write(configfile)
       gen_config = dict(config.items('GENERATION_CONFIG')) if config.has_section('GENERATION_CONFIG') else {}
       safety_settings = dict(config.items('SAFETY_SETTINGS')) if config.has_section('SAFETY_SETTINGS') else {}
       model_name = config.get('MODEL', 'model_name', fallback='gemini-1.5-flash-latest')
       time_delay = config.getfloat('TIME_DELAY', 'time_delay', fallback=1.2)
       prompt_template = config.get('MESSAGES', 'translate_text_template', fallback="{audio_prompt}")
       log_message(log_widget, f"Using Gemini Key: {current_key_name}", level="INFO")
       return selected_key, gen_config, safety_settings, model_name, time_delay, prompt_template
   except configparser.Error as e:
       log_message(log_widget, f"Error reading Gemini settings file {config_path}: {e}", level="ERROR")
       return None, None, None, None, 1.2, "{audio_prompt}"
   except Exception as e:
        log_message(log_widget, f"Unexpected error processing Gemini settings {config_path}: {e}", level="CRITICAL")
        return None, None, None, None, 1.2, "{audio_prompt}"
# ===============================================
# ENHANCED SEARCHABLE COMBOBOX
# ===============================================
class SearchableComboBox(ctk.CTkComboBox):
   """Enhanced ComboBox with search functionality"""
   def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self._original_values = []
       self._search_var = tk.StringVar()
       self._search_var.trace('w', self._on_search)
   def configure(self, **kwargs):
       if 'values' in kwargs:
           self._original_values = kwargs['values'][:]
       super().configure(**kwargs)
   def set_values(self, values):
       """Set values for the combobox"""
       self._original_values = values[:]
       self.configure(values=values)
   def _on_search(self, *args):
       """Filter values based on search text"""
       search_text = self._search_var.get().lower()
       if not search_text:
           filtered_values = self._original_values
       else:
           filtered_values = [val for val in self._original_values if search_text in val.lower()]
       self.configure(values=filtered_values)
       # Auto-select first match if available
       if filtered_values and search_text:
           current_val = self.get()
           if current_val not in filtered_values:
               self.set(filtered_values[0])
   def bind_search(self, widget):
       """Bind search functionality to the entry widget"""
       widget.configure(textvariable=self._search_var)
# ===============================================
# FREQUENCY ANALYSIS TASK
# ===============================================
def analyze_frequencies_task(input_dir, output_dir, start_time_s, end_time_s, filter_non_speech, status_callback, progress_callback, log_widget, auto_shutdown=False, male_min=None, male_max=None, female_min=None, female_max=None):
   global gender_analysis_results
   gender_analysis_results.clear()
   # Start time tracking
   task_name = "frequency_analysis"
   start_task_timer(task_name, output_dir)
   # Load custom frequency ranges if provided
   if male_min is None:
       male_min, male_max, female_min, female_max = load_gender_frequency_settings()
   # Load existing progress
   progress_data = load_progress(output_dir, FREQUENCY_PROGRESS_FILE)
   processed_files = set()
   if progress_data:
       processed_files = set(progress_data.get('processed_files', []))
       log_message(log_widget, f"Resuming frequency analysis. Previously processed: {len(processed_files)} files", level="INFO")
   files_to_process = []
   status_callback("Listing files...")
   try:
       for filename in os.listdir(input_dir):
           if stop_processing_flag.is_set():
               status_callback("Frequency analysis cancelled.")
               stop_task_timer(task_name, output_dir)
               return
           full_path = os.path.join(input_dir, filename)
           if os.path.isfile(full_path) and any(filename.lower().endswith(ext) for ext in SUPPORTED_AUDIO_EXTENSIONS):
               if filename not in processed_files:
                   files_to_process.append(filename)
   except Exception as e:
       log_message(log_widget, f"Error listing files in {input_dir}: {e}", level="ERROR")
       status_callback(f"Error: {e}")
       stop_task_timer(task_name, output_dir)
       return
   total_files = len(files_to_process) + len(processed_files)
   if total_files == 0:
       log_message(log_widget, "No supported audio files found for frequency analysis.", level="WARNING")
       status_callback("No audio files found.")
       stop_task_timer(task_name, output_dir)
       return
   output_file_path = os.path.join(output_dir, OUTPUT_ANALYSIS_FILE)
   processed_count = len(processed_files)
   try:
       # Open file in append mode if resuming
       mode = "a" if progress_data else "w"
       with open(output_file_path, mode, encoding="utf-8") as outfile:
           if not progress_data:  # Write header only for new files
               outfile.write("Filename|EstimatedGender|AvgF0(Hz)|StartTime(s)|EndTime(s)\n")
           for i, filename in enumerate(files_to_process):
               if stop_processing_flag.is_set():
                   status_callback(f"Frequency analysis cancelled after {processed_count}/{total_files} files.")
                   log_message(log_widget, "Frequency analysis cancelled by user.", level="INFO")
                   stop_task_timer(task_name, output_dir)
                   return
               progress = (processed_count) / total_files
               progress_callback(progress)
               # Update time tracking
               total_time, session_time = update_task_timer(task_name, output_dir)
               estimated_remaining = estimate_completion_time(progress, total_time) if progress > 0 else 0
               time_status = f"Time: {format_time_duration(total_time)} | ETA: {format_time_duration(estimated_remaining)}"
               status_callback(f"Analyzing {processed_count+1}/{total_files}: {filename} | {time_status}")
               log_message(log_widget, f"Analyzing frequency for: {filename}", level="DEBUG")
               file_path = os.path.join(input_dir, filename)
               f0_mean_overall = None
               estimated_gender = "Unknown"
               try:
                   y, sr = librosa.load(file_path, sr=None, offset=start_time_s, duration=(end_time_s - start_time_s) if end_time_s > start_time_s else None)
                   if len(y) == 0:
                        log_message(log_widget, f"Skipping {filename}: No audio data in specified range.", level="WARNING")
                        outfile.write(f"{filename}|NoAudioDataInRange|N/A|{start_time_s}|{end_time_s}\n")
                        gender_analysis_results[filename] = "Unknown"
                        processed_count += 1
                        progress = processed_count / total_files
                        progress_callback(progress)
                        continue
                   f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                   f0_voiced = f0[voiced_flag]
                   if filter_non_speech and len(f0_voiced) < 10:
                        log_message(log_widget, f"Skipping {filename}: Filtered as likely non-speech.", level="INFO")
                        outfile.write(f"{filename}|FilteredNonSpeech|N/A|{start_time_s}|{end_time_s}\n")
                        gender_analysis_results[filename] = "Unknown"
                        processed_count += 1
                        progress = processed_count / total_files
                        progress_callback(progress)
                        continue
                   if len(f0_voiced) > 0:
                       f0_mean_overall = np.nanmean(f0_voiced)
                       if np.isnan(f0_mean_overall):
                           f0_mean_overall = None
                           estimated_gender = "Unknown"
                           log_message(log_widget, f"Could not calculate mean F0 for {filename}", level="WARNING")
                       else:
                           estimated_gender = estimate_gender_from_f0_custom(f0_mean_overall, male_min, male_max, female_min, female_max)
                           log_message(log_widget, f"{filename}: Avg F0 = {f0_mean_overall:.2f} Hz, Estimated Gender = {estimated_gender}", level="DEBUG")
                   else:
                       log_message(log_widget, f"No voiced frames detected for F0 calculation in {filename}", level="WARNING")
                       estimated_gender = "Unknown"
                   f0_display = f"{f0_mean_overall:.2f}" if f0_mean_overall is not None else "N/A"
                   outfile.write(f"{filename}|{estimated_gender}|{f0_display}|{start_time_s}|{end_time_s}\n")
                   gender_analysis_results[filename] = estimated_gender
                   processed_count += 1
                   # Save progress
                   processed_files.add(filename)
                   progress_data = {
                       'processed_files': list(processed_files),
                       'total_files': total_files,
                       'last_updated': datetime.datetime.now().isoformat()
                   }
                   save_progress(output_dir, FREQUENCY_PROGRESS_FILE, progress_data)
               except Exception as e:
                   log_message(log_widget, f"Error processing {filename}: {e}", level="ERROR")
                   outfile.write(f"{filename}|ErrorProcessing|{e}|{start_time_s}|{end_time_s}\n")
                   gender_analysis_results[filename] = "Error"
                   processed_count += 1
           final_time = stop_task_timer(task_name, output_dir)
           status_callback(f"Frequency analysis complete. Processed {processed_count}/{total_files} files. Total time: {format_time_duration(final_time)}")
           log_message(log_widget, f"Frequency analysis finished. Results saved to {output_file_path}", level="INFO")
           # Clear progress file on completion
           clear_progress(output_dir, FREQUENCY_PROGRESS_FILE)
           # Auto shutdown if requested
           if auto_shutdown:
               log_message(log_widget, "Auto shutdown requested. Shutting down Windows in 30 seconds...", level="INFO")
               shutdown_windows()
   except Exception as e:
       log_message(log_widget, f"Failed to write analysis results to {output_file_path}: {e}", level="CRITICAL")
       status_callback(f"Error writing results: {e}")
       stop_task_timer(task_name, output_dir)
# ===============================================
# TRANSCRIPTION TASK
# ===============================================
def transcribe_files_task(input_dir, output_dir, engine, options, status_callback, progress_callback, log_widget, auto_shutdown=False):
   # --- START: ส่วนที่แก้ไขเพิ่มเติม ---
   # 1. สร้างโฟลเดอร์สำหรับไฟล์ที่ไม่มีเสียงพูด
   non_speech_dir = os.path.join(output_dir, "Non_speech")
   os.makedirs(non_speech_dir, exist_ok=True)
   # 2. อ่านข้อมูลการวิเคราะห์ความถี่จาก Analyzed_Sounds.txt มาเก็บไว้
   analysis_data = {}
   analysis_file_path = os.path.join(output_dir, OUTPUT_ANALYSIS_FILE)
   try:
       with open(analysis_file_path, 'r', encoding='utf-8') as f:
           lines = f.readlines()
           # ข้าม header
           for line in lines[1:]:
               parts = line.strip().split('|')
               if len(parts) >= 3:
                   filename = parts[0]
                   # จัดเก็บ prefix ในรูปแบบ Filename|EstimatedGender|AvgF0(Hz)
                   prefix = '|'.join(parts[:3])
                   analysis_data[filename] = prefix
       log_message(log_widget, f"Successfully loaded data from {OUTPUT_ANALYSIS_FILE}", level="INFO")
   except FileNotFoundError:
       log_message(log_widget, f"'{OUTPUT_ANALYSIS_FILE}' not found. Gender and F0 data will be unavailable.", level="WARNING")
   except Exception as e:
       log_message(log_widget, f"Error reading '{OUTPUT_ANALYSIS_FILE}': {e}", level="ERROR")
   # --- END: ส่วนที่แก้ไขเพิ่มเติม ---
   # Start time tracking
   task_name = "transcription"
   start_task_timer(task_name, output_dir)
   # Load existing progress
   progress_data = load_progress(output_dir, TRANSCRIPTION_PROGRESS_FILE)
   processed_files = set()
   if progress_data:
       processed_files = set(progress_data.get('processed_files', []))
       log_message(log_widget, f"Resuming transcription. Previously processed: {len(processed_files)} files", level="INFO")
   files_to_process = []
   status_callback("Listing files for transcription...")
   try:
       for filename in os.listdir(input_dir):
           if stop_processing_flag.is_set():
               status_callback("Transcription cancelled.")
               stop_task_timer(task_name, output_dir)
               return
           full_path = os.path.join(input_dir, filename)
           if os.path.isfile(full_path) and any(filename.lower().endswith(ext) for ext in SUPPORTED_INPUT_EXTENSIONS):
               if filename not in processed_files:
                   files_to_process.append(filename)
   except Exception as e:
       log_message(log_widget, f"Error listing files in {input_dir}: {e}", level="ERROR")
       status_callback(f"Error: {e}")
       stop_task_timer(task_name, output_dir)
       return
   total_files = len(files_to_process) + len(processed_files)
   if total_files == 0:
       log_message(log_widget, "No supported files found for transcription.", level="WARNING")
       status_callback("No input files found.")
       stop_task_timer(task_name, output_dir)
       return
   output_file_path = os.path.join(output_dir, OUTPUT_TRANSCRIPTION_FILE)
   processed_count = len(processed_files)
   gemini_client = None
   whisper_model = None
   # Initialize engines
   if engine == "Whisper":
       if not WHISPER_AVAILABLE:
           log_message(log_widget, "Whisper selected but library not available.", level="ERROR")
           status_callback("Error: Whisper library not installed.")
           stop_task_timer(task_name, output_dir)
           return
       model_setting = options.get('whisper_model', 'base')
       status_callback(f"Setting up Whisper environment...")
       if os.path.exists(model_setting):
           whisper_model = safe_whisper_load_model_from_file(model_setting, log_widget)
           if whisper_model is None:
               log_message(log_widget, f"Failed to load Whisper model from file '{model_setting}'.", level="ERROR")
               status_callback(f"Error: Cannot load Whisper model from file")
               stop_task_timer(task_name, output_dir)
               return
           status_callback(f"Whisper model loaded from file successfully.")
       else:
           whisper_model = safe_whisper_load_model(model_setting, log_widget)
           if whisper_model is None:
               log_message(log_widget, f"Failed to load Whisper model '{model_setting}'. Please check internet connection and try again.", level="ERROR")
               status_callback(f"Error: Cannot load Whisper model '{model_setting}'")
               stop_task_timer(task_name, output_dir)
               return
           status_callback(f"Whisper model '{model_setting}' ready for transcription.")
   elif engine == "Google Cloud STT":
       if not GOOGLE_STT_AVAILABLE:
           log_message(log_widget, "Google Cloud STT selected but library not available.", level="ERROR")
           status_callback("Error: Google Cloud Speech library not installed.")
           stop_task_timer(task_name, output_dir)
           return
       key_path = options.get('google_key_path')
       if not key_path or not os.path.exists(key_path):
           log_message(log_widget, "Google STT key file path not set or file not found.", level="ERROR")
           status_callback("Error: Google STT Key file not configured.")
           stop_task_timer(task_name, output_dir)
           return
       try:
           stt_client = speech.SpeechClient.from_service_account_json(key_path)
           log_message(log_widget, "Google Cloud STT client initialized.", level="INFO")
       except Exception as e:
           log_message(log_widget, f"Failed to initialize Google Cloud STT client: {e}", level="ERROR")
           status_callback(f"Error initializing Google STT: {e}")
           stop_task_timer(task_name, output_dir)
           return
   elif engine == "Gemini":
       if not GEMINI_AVAILABLE:
           log_message(log_widget, "Gemini selected but library not available.", level="ERROR")
           status_callback("Error: Gemini library not installed.")
           stop_task_timer(task_name, output_dir)
           return
       settings_path = options.get('gemini_settings_path', DEFAULT_GEMINI_SETTINGS_FILE)
       api_key, gen_config, safety_settings, model_name_from_ini, time_delay, prompt_template = get_gemini_api_key_from_ini(settings_path, log_widget)
       if not api_key:
           status_callback("Error: Could not get valid Gemini API key.")
           stop_task_timer(task_name, output_dir)
           return
       model_name = options.get('gemini_model_name') or model_name_from_ini or 'gemini-1.5-flash-latest'
       try:
           genai.configure(api_key=api_key)
           gemini_client = genai.GenerativeModel(
               model_name=model_name,
               generation_config=gen_config,
               safety_settings=safety_settings
           )
           log_message(log_widget, f"Gemini client initialized with model: {model_name}", level="INFO")
       except Exception as e:
           log_message(log_widget, f"Failed to initialize Gemini client with model {model_name}: {e}", level="ERROR")
           status_callback(f"Error initializing Gemini: {e}")
           stop_task_timer(task_name, output_dir)
           return
   # Process files
   try:
       # Open file in append mode if resuming
       mode = "a" if progress_data else "w"
       with open(output_file_path, mode, encoding="utf-8") as outfile:
           for i, filename in enumerate(files_to_process):
               if stop_processing_flag.is_set():
                   status_callback(f"Transcription cancelled after {processed_count}/{total_files} files.")
                   log_message(log_widget, "Transcription cancelled by user.", level="INFO")
                   stop_task_timer(task_name, output_dir)
                   return
               progress = (processed_count) / total_files
               progress_callback(progress)
               # Update time tracking
               total_time, session_time = update_task_timer(task_name, output_dir)
               estimated_remaining = estimate_completion_time(progress, total_time) if progress > 0 else 0
               time_status = f"Time: {format_time_duration(total_time)} | ETA: {format_time_duration(estimated_remaining)}"
               status_callback(f"Transcribing {processed_count+1}/{total_files}: {filename} | {time_status}")
               log_message(log_widget, f"Starting transcription for: {filename}", level="DEBUG")
               file_path = os.path.join(input_dir, filename)
               transcript = "Transcription Error"
               detected_language = options.get('language', 'en')
               try:
                   # Whisper Engine
                   if engine == "Whisper" and whisper_model:
                       result = whisper_model.transcribe(file_path, language=detected_language if detected_language != 'auto' else None)
                       transcript = result["text"].strip()
                       if detected_language == 'auto':
                           detected_language = result.get("language", "auto")
                       log_message(log_widget, f"Whisper transcription success for {filename}. Language: {detected_language}", level="DEBUG")
                   # Google Cloud STT Engine
                   elif engine == "Google Cloud STT" and 'stt_client' in locals():
                       try:
                           status_callback(f"Converting {filename} for Google STT...")
                           audio = AudioSegment.from_file(file_path)
                           temp_wav_path = os.path.join(output_dir, f"_temp_{os.path.splitext(filename)[0]}.wav")
                           audio.export(temp_wav_path, format="wav")
                           with open(temp_wav_path, "rb") as audio_file:
                               content = audio_file.read()
                           with sf.SoundFile(temp_wav_path) as sf_file:
                               sample_rate = sf_file.samplerate
                               channels = sf_file.channels
                               if channels > 1:
                                   log_message(log_widget, f"Warning: {filename} has {channels} channels, Google STT works best with mono. Using converted mono.", level="WARNING")
                           recognition_config = speech.RecognitionConfig(
                               encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                               sample_rate_hertz=sample_rate,
                               language_code=detected_language if detected_language != 'auto' else "en-US",
                               enable_automatic_punctuation=True,
                           )
                           audio_input = speech.RecognitionAudio(content=content)
                           status_callback(f"Sending {filename} to Google STT API...")
                           response = stt_client.recognize(config=recognition_config, audio=audio_input)
                           status_callback(f"Processing Google STT response for {filename}...")
                           if response.results:
                               transcript = response.results[0].alternatives[0].transcript.strip()
                               log_message(log_widget, f"Google STT transcription success for {filename}", level="DEBUG")
                           else:
                               transcript = "" # No speech detected, use empty string for filtering
                               log_message(log_widget, f"Google STT: No speech detected in {filename}", level="WARNING")
                           os.remove(temp_wav_path)
                       except Exception as e:
                           transcript = f"Google STT Processing Error: {e}"
                           log_message(log_widget, f"Error during Google STT processing for {filename}: {e}", level="ERROR")
                           if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
                               try: 
                                   os.remove(temp_wav_path)
                               except: 
                                   pass
                   # Gemini Engine
                   elif engine == "Gemini" and gemini_client:
                       status_callback(f"Preparing {filename} for Gemini API...")
                       try:
                           audio_file_for_gemini = genai.upload_file(path=file_path)
                           log_message(log_widget, f"Uploaded {filename} to Gemini.", level="DEBUG")
                           prompt = options.get("gemini_prompt", f"Transcribe the following audio file accurately. The language is likely {detected_language if detected_language != 'auto' else 'unknown, please detect'}.")
                           status_callback(f"Sending {filename} to Gemini API...")
                           response = gemini_client.generate_content([prompt, audio_file_for_gemini])
                           status_callback(f"Processing Gemini response for {filename}...")
                           transcript = response.text.strip()
                           log_message(log_widget, f"Gemini transcription success for {filename}.", level="DEBUG")
                           time.sleep(time_delay)
                       except Exception as e:
                           transcript = f"Gemini Processing Error: {e}"
                           log_message(log_widget, f"Error during Gemini transcription for {filename}: {e}", level="ERROR")
                           if 'audio_file_for_gemini' in locals() and hasattr(audio_file_for_gemini, 'name'):
                               try:
                                   genai.delete_file(audio_file_for_gemini.name)
                               except Exception: 
                                   pass
                   # Other Engines (Not Implemented)
                   elif engine == "Copilot":
                       transcript = "Copilot transcription not implemented."
                       log_message(log_widget, f"Skipping {filename}: Copilot transcription selected but not implemented.", level="WARNING")
                   elif engine == "ChatGPT":
                       transcript = "ChatGPT transcription not implemented."
                       log_message(log_widget, f"Skipping {filename}: ChatGPT transcription selected but not implemented.", level="WARNING")
                   else:
                       transcript = "Unsupported transcription engine selected."
                       log_message(log_widget, f"Skipping {filename}: Unsupported engine '{engine}'.", level="ERROR")
               except Exception as e:
                   transcript = f"General Transcription Error: {e}"
                   log_message(log_widget, f"Unexpected error transcribing {filename}: {e}", level="ERROR")
               # --- START: ส่วนที่แก้ไขเพิ่มเติม ---
               # 3. ตรวจสอบว่ามีเสียงพูดหรือไม่
               # ให้ถือว่าเป็นเสียงพูดถ้ามีตัวอักษรหรือตัวเลขอยู่
               is_speech = transcript and transcript.strip() and any(c.isalnum() for c in transcript)
               if not is_speech:
                   log_message(log_widget, f"File '{filename}' contains no speech. Moving to Non_speech folder.", level="INFO")
                   source_path = os.path.join(input_dir, filename)
                   dest_path = os.path.join(non_speech_dir, filename)
                   try:
                       if os.path.exists(source_path):
                           shutil.move(source_path, dest_path)
                           log_message(log_widget, f"Moved {filename} to {dest_path}", level="DEBUG")
                       else:
                           log_message(log_widget, f"Source file {source_path} not found for moving.", level="WARNING")
                   except Exception as move_error:
                       log_message(log_widget, f"Could not move {filename} to Non_speech folder: {move_error}", level="ERROR")
                   # ข้ามการเขียนไฟล์นี้ลงใน OriginalSounds.txt แต่ยังคงนับว่าประมวลผลแล้ว
                   processed_count += 1
               else:
                   # 4. ถ้ามีเสียงพูด ให้เขียนผลลัพธ์โดยใช้ข้อมูลจาก analysis
                   # ดึง prefix (Filename|Gender|F0) จากข้อมูลที่อ่านไว้
                   prefix = analysis_data.get(filename)
                   if not prefix:
                       # สร้าง prefix สำรองในกรณีที่ไม่มีข้อมูล analysis
                       estimated_gender = gender_analysis_results.get(filename, "Unknown")
                       prefix = f"{filename}|{estimated_gender}|N/A"
                       log_message(log_widget, f"Could not find frequency analysis data for '{filename}'. Using fallback.", level="WARNING")
                   # สร้างบรรทัดผลลัพธ์ในรูปแบบ: Filename|EstimatedGender|AvgF0(Hz)|Transcript
                   output_line = f"{prefix}|{transcript}"
                   outfile.write(output_line + "\n")
                   processed_count += 1
               # --- END: ส่วนที่แก้ไขเพิ่มเติม ---
               # Save progress (ย้ายมาอยู่นอก if/else เพื่อให้บันทึก progress ทั้งสองกรณี)
               processed_files.add(filename)
               progress_data = {
                   'processed_files': list(processed_files),
                   'total_files': total_files,
                   'engine': engine,
                   'options': options,
                   'last_updated': datetime.datetime.now().isoformat()
               }
               save_progress(output_dir, TRANSCRIPTION_PROGRESS_FILE, progress_data)
           final_time = stop_task_timer(task_name, output_dir)
           status_callback(f"Transcription complete. Processed {processed_count}/{total_files} files. Total time: {format_time_duration(final_time)}")
           log_message(log_widget, f"Transcription finished. Results saved to {output_file_path}", level="INFO")
           # Clear progress file on completion
           clear_progress(output_dir, TRANSCRIPTION_PROGRESS_FILE)
           # Auto shutdown if requested
           if auto_shutdown:
               log_message(log_widget, "Auto shutdown requested. Shutting down Windows in 30 seconds...", level="INFO")
               shutdown_windows()
   except Exception as e:
       log_message(log_widget, f"Failed to write transcription results to {output_file_path}: {e}", level="CRITICAL")
       status_callback(f"Error writing results: {e}")
       stop_task_timer(task_name, output_dir)
# ===============================================
# TRANSLATION TASK
# ===============================================
def translate_texts_task(output_dir, target_language, status_callback, progress_callback, log_widget, auto_shutdown=False):
   input_file_path = os.path.join(output_dir, OUTPUT_TRANSCRIPTION_FILE)
   output_file_path = os.path.join(output_dir, OUTPUT_TRANSLATION_FILE)
   if not os.path.exists(input_file_path):
       log_message(log_widget, f"Transcription file not found: {input_file_path}", level="ERROR")
       status_callback("Error: OriginalSounds.txt not found.")
       return
   # Load existing progress
   progress_data = load_progress(output_dir, TRANSLATION_PROGRESS_FILE)
   processed_lines = 0
   if progress_data:
       processed_lines = progress_data.get('processed_lines', 0)
       log_message(log_widget, f"Resuming translation. Previously processed: {processed_lines} lines", level="INFO")
   lines_to_translate = []
   try:
       with open(input_file_path, "r", encoding="utf-8") as infile:
           lines_to_translate = infile.readlines()
   except Exception as e:
       log_message(log_widget, f"Error reading transcription file {input_file_path}: {e}", level="ERROR")
       status_callback(f"Error reading input file: {e}")
       return
   total_lines = len(lines_to_translate)
   if total_lines == 0:
       log_message(log_widget, "Transcription file is empty. Nothing to translate.", level="WARNING")
       status_callback("Input file is empty.")
       return
   status_callback("Initializing translator...")
   try:
       translator = Translator()
       log_message(log_widget, "Googletrans Translator initialized.", level="INFO")
   except Exception as e:
        log_message(log_widget, f"Failed to initialize translator: {e}", level="ERROR")
        status_callback(f"Error init translator: {e}")
        return
   try:
       # Open file in append mode if resuming
       mode = "a" if progress_data and processed_lines > 0 else "w"
       with open(output_file_path, mode, encoding="utf-8") as outfile:
           # Skip already processed lines if resuming
           lines_to_process = lines_to_translate[processed_lines:] if processed_lines > 0 else lines_to_translate
           for i, line in enumerate(lines_to_process):
               current_line_index = processed_lines + i
               if stop_processing_flag.is_set():
                   status_callback(f"Translation cancelled after {current_line_index + 1}/{total_lines} lines.")
                   log_message(log_widget, "Translation cancelled by user.", level="INFO")
                   return
               progress = (current_line_index + 1) / total_lines
               progress_callback(progress)
               filename, gender, age, channel, original_text = parse_output_line(line)
               if original_text is None or not original_text.strip():
                   log_message(log_widget, f"Skipping empty line or parse error at line {current_line_index + 1}", level="WARNING")
                   outfile.write(line)
                   continue
               status_callback(f"Translating line {current_line_index + 1}/{total_lines} to '{target_language}'...")
               translated_text = original_text
               try:
                   if original_text.strip():
                       translation_result = translator.translate(original_text, dest=target_language)
                       translated_text = translation_result.text
                       log_message(log_widget, f"Translated line {current_line_index + 1}: '{original_text[:30]}...' -> '{translated_text[:30]}...'", level="DEBUG")
                       time.sleep(0.1)
                   else:
                        log_message(log_widget, f"Skipping translation for empty text on line {current_line_index + 1}", level="DEBUG")
               except Exception as e:
                   translated_text = f"Translation Error: {e}"
                   log_message(log_widget, f"Error translating line {current_line_index + 1} ('{original_text[:50]}...'): {e}", level="WARNING")
               output_line = format_output_line(filename if filename else f"Line_{current_line_index + 1}", gender, age, channel, translated_text)
               outfile.write(output_line + "\n")
               # Save progress
               progress_data = {
                   'processed_lines': current_line_index + 1,
                   'total_lines': total_lines,
                   'target_language': target_language,
                   'last_updated': datetime.datetime.now().isoformat()
               }
               save_progress(output_dir, TRANSLATION_PROGRESS_FILE, progress_data)
           status_callback(f"Translation complete. Processed {total_lines}/{total_lines} lines.")
           log_message(log_widget, f"Translation finished. Results saved to {output_file_path}", level="INFO")
           # Clear progress file on completion
           clear_progress(output_dir, TRANSLATION_PROGRESS_FILE)
           # Auto shutdown if requested
           if auto_shutdown:
               log_message(log_widget, "Auto shutdown requested. Shutting down Windows in 30 seconds...", level="INFO")
               shutdown_windows()
   except Exception as e:
       log_message(log_widget, f"Failed to write translation results to {output_file_path}: {e}", level="CRITICAL")
       status_callback(f"Error writing results: {e}")
# ===============================================
# ENHANCED SPEECH SYNTHESIS WITH EMOTION ANALYSIS
# ===============================================
def synthesize_single_chunk(text, output_path, engine, config, output_format, log_widget, google_tts_client=None, emotion_analyzer=None, ssml_generator=None, use_auto_emotion=False, use_advanced_emotion=False, quota_manager=None):
    """สร้างเสียงสำหรับข้อความส่วนเดียว พร้อมระบบวิเคราะห์อารมณ์อัตโนมัติ, gTTS delay และ quota management"""
    try:
        if engine == "Google Cloud TTS":
            # ส่ง key_path ที่ได้จาก config ของ channel นั้นๆ ไปด้วย
            return synthesize_google_tts_chunk(text, output_path, config, output_format, log_widget, google_tts_client, emotion_analyzer, ssml_generator, use_auto_emotion, use_advanced_emotion, quota_manager)
        elif engine == "gTTS":
            request_delay = config.get('request_delay', 0.0)
            return synthesize_gtts_chunk(text, output_path, config, output_format, log_widget, request_delay)
        else:
            log_message(log_widget, f"Unsupported synthesis engine: {engine}", level="ERROR")
            return False
    except Exception as e:
        log_message(log_widget, f"Error in synthesize_single_chunk: {e}", level="ERROR")
        return False
def synthesize_google_tts_chunk(text, output_path, config, output_format, log_widget, google_tts_client, emotion_analyzer=None, ssml_generator=None, use_auto_emotion=False, use_advanced_emotion=False, quota_manager=None):
    """สร้างเสียงด้วย Google Cloud TTS พร้อมระบบวิเคราะห์อารมณ์และ quota management (เวอร์ชันปรับปรุงสำหรับหลาย key)"""
    try:
        key_path = config.get('google_key_path')
        if not key_path or not os.path.exists(key_path):
            raise ValueError(f"Google TTS key file not found for this channel: {key_path}")
        current_tts_client = google_tts_client
        if not current_tts_client:
            current_tts_client = texttospeech.TextToSpeechClient.from_service_account_json(key_path)
        # Configure synthesis parameters
        lang_code = config.get('languageCode', 'th-TH')
        voice_name = config.get('name', 'th-TH-Standard-A')
        ssml_gender_str = config.get('ssmlGender', 'NEUTRAL').upper()
        speaking_rate = config.get('speakingRate', 1.0)
        pitch = config.get('pitch', 0.0)
        volume_db = config.get('volumeGainDb', 0.0)
        emotion_style = config.get('emotion_style', 'neutral')
        custom_ssml = config.get('custom_ssml', '')
        # === QUOTA CHECK AND UPDATE (ใช้ key_path) ===
        if quota_manager:
            can_proceed, quota_info = quota_manager.check_quota(text, voice_name, key_path)
            if not can_proceed:
                error_msg = handle_quota_exceeded_error(quota_info, voice_name, log_widget)
                log_message(log_widget, f"Quota exceeded for key {os.path.basename(key_path)}: synthesis skipped", level="ERROR")
                raise Exception(f"Quota Exceeded: {error_msg}")
            voice_type = quota_info.get('voice_type', 'unknown')
            current_usage = quota_info.get('current_usage', 0)
            requested_usage = quota_info.get('requested_usage', 0)
            unit = quota_info.get('unit', 'characters')
            log_message(log_widget, f"Key '{os.path.basename(key_path)}' Quota check passed - {voice_type}: {current_usage:,}+{requested_usage:,} {unit}", level="DEBUG")
        # (ส่วนที่เหลือของฟังก์ชันเหมือนเดิม)
        # Set gender
        ssml_gender_enum = texttospeech.SsmlVoiceGender.NEUTRAL
        if ssml_gender_str == 'MALE':
            ssml_gender_enum = texttospeech.SsmlVoiceGender.MALE
        elif ssml_gender_str == 'FEMALE':
            ssml_gender_enum = texttospeech.SsmlVoiceGender.FEMALE
        # === EMOTION ANALYSIS AND SSML PREPARATION ===
        synthesis_input = texttospeech.SynthesisInput()
        thai_emotion_voices = ['th-TH-Neural2-C', 'th-TH-Standard-A']
        voice_supports_emotion = any(supported_voice in voice_name for supported_voice in thai_emotion_voices)
        if use_auto_emotion and emotion_analyzer and ssml_generator and voice_supports_emotion:
            try:
                auto_ssml, analysis = create_auto_emotion_ssml(text, emotion_analyzer, ssml_generator, use_advanced_emotion)
                synthesis_input.ssml = auto_ssml
                detected_emotion = analysis.get('emotion', 'neutral')
                analysis_mode = "advanced" if use_advanced_emotion else "simple"
                log_message(log_widget, f"Auto emotion ({analysis_mode}): {detected_emotion} for text: {text[:50]}...", level="DEBUG")
            except Exception as e:
                log_message(log_widget, f"Error in auto emotion analysis, falling back to manual: {e}", level="WARNING")
                use_auto_emotion = False
        if not use_auto_emotion:
            if voice_supports_emotion:
                if emotion_style == "custom" and custom_ssml.strip():
                    synthesis_input.ssml = custom_ssml.replace("{text}", text)
                    log_message(log_widget, f"Using custom SSML for {voice_name}", level="DEBUG")
                elif emotion_style != "neutral":
                    emotion_ssml = create_emotion_ssml(text, emotion_style)
                    synthesis_input.ssml = emotion_ssml
                    log_message(log_widget, f"Using manual emotion '{emotion_style}' for {voice_name}", level="DEBUG")
                elif text.strip().startswith("<speak>") and text.strip().endswith("</speak>"):
                    synthesis_input.ssml = text
                else:
                    synthesis_input.text = text
            else:
                if text.strip().startswith("<speak>") and text.strip().endswith("</speak>"):
                    import re
                    plain_text = re.sub(r'<[^>]+>', '', text)
                    synthesis_input.text = plain_text
                    log_message(log_widget, f"Voice {voice_name} doesn't support SSML, using plain text", level="WARNING")
                else:
                    synthesis_input.text = text
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name,
            ssml_gender=ssml_gender_enum
        )
        audio_encoding_enum = texttospeech.AudioEncoding.LINEAR16
        if output_format == 'mp3':
            audio_encoding_enum = texttospeech.AudioEncoding.MP3
        elif output_format == 'ogg':
            audio_encoding_enum = texttospeech.AudioEncoding.OGG_OPUS
        audio_config = texttospeech.AudioConfig(
            audio_encoding=audio_encoding_enum,
            speaking_rate=speaking_rate,
            pitch=pitch,
            volume_gain_db=volume_db
        )
        # === SYNTHESIZE ===
        log_message(log_widget, f"Synthesizing with Google TTS: {voice_name} (Key: {os.path.basename(key_path)})", level="DEBUG")
        response = current_tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open(output_path, "wb") as out_audio:
            out_audio.write(response.audio_content)
        # === UPDATE QUOTA AFTER SUCCESSFUL SYNTHESIS (ใช้ key_path) ===
        if quota_manager:
            try:
                usage_info = quota_manager.update_usage(text, voice_name, key_path)
                voice_type = usage_info.get('voice_type', 'unknown')
                usage_added = usage_info.get('usage_added', 0)
                total_usage = usage_info.get('total_usage', 0)
                unit = usage_info.get('unit', 'characters')
                log_message(log_widget, f"Quota updated for key '{os.path.basename(key_path)}' - {voice_type}: +{usage_added:,} {unit}, total: {total_usage:,} {unit}", level="DEBUG")
            except Exception as quota_error:
                log_message(log_widget, f"Error updating quota for key {os.path.basename(key_path)}: {quota_error}", level="WARNING")
        log_message(log_widget, f"Google TTS synthesis successful: {os.path.basename(output_path)}", level="DEBUG")
        return True
    except Exception as e:
        log_message(log_widget, f"Google TTS synthesis failed: {e}", level="ERROR")
        return False
def synthesize_gtts_chunk(text, output_path, config, output_format, log_widget, request_delay=0):
    """สร้างเสียงด้วย gTTS พร้อม delay"""
    try:
        lang_code = config.get('languageCode', 'th')
        speed_factor = config.get('speed', 1.0)
        pitch_semitones = config.get('pitch', 0.0)
        # Apply request delay
        if request_delay > 0:
            log_message(log_widget, f"gTTS request delay: {request_delay} seconds", level="DEBUG")
            time.sleep(request_delay)
        tts = gTTS(text=text, lang=lang_code, slow=False)
        temp_base_path = os.path.join(os.path.dirname(output_path), f"_temp_gtts_{os.getpid()}.mp3")
        tts.save(temp_base_path)
        # Apply audio effects
        audio = AudioSegment.from_mp3(temp_base_path)
        final_audio = audio
        if speed_factor != 1.0:
            final_audio = audio.speedup(playback_speed=speed_factor)
        if pitch_semitones != 0.0:
            new_frame_rate = int(final_audio.frame_rate * (2**(pitch_semitones/12.0)))
            final_audio = final_audio._spawn(final_audio.raw_data, overrides={'frame_rate': new_frame_rate})
        final_audio.export(output_path, format=output_format)
        os.remove(temp_base_path)
        log_message(log_widget, f"gTTS synthesis successful: {os.path.basename(output_path)}", level="DEBUG")
        return True
    except Exception as e:
        log_message(log_widget, f"gTTS synthesis failed: {e}", level="ERROR")
        if 'temp_base_path' in locals() and os.path.exists(temp_base_path):
            try:
                os.remove(temp_base_path)
            except:
                pass
        return False
def enhanced_synthesize_speech_task(input_text_file, output_dir, engine, channel_configs, output_format, status_callback, progress_callback, log_widget, auto_shutdown, regex_mode, custom_regex, use_auto_emotion=False, use_advanced_emotion=False):
    """Enhanced speech synthesis with text splitting support, emotion analysis, quota management, and gTTS delay"""
    # Start time tracking
    task_name = "synthesis"
    start_task_timer(task_name, output_dir)
    if not os.path.exists(input_text_file):
        log_message(log_widget, f"Input text file not found: {input_text_file}", level="ERROR")
        status_callback(f"Error: Input file not found.")
        stop_task_timer(task_name, output_dir)
        return
    # Initialize emotion system if requested
    emotion_analyzer = None
    ssml_generator = None
    if use_auto_emotion:
        try:
            emotion_analyzer = EmotionAnalyzer()
            ssml_generator = SSMLGenerator(emotion_analyzer)
            log_message(log_widget, "Emotion analysis system initialized", level="INFO")
        except Exception as e:
            log_message(log_widget, f"Failed to initialize emotion system: {e}, continuing without auto emotion", level="WARNING")
            use_auto_emotion = False
    # Initialize quota management system
    quota_manager = create_quota_manager(output_dir)
    if quota_manager:
        log_message(log_widget, "Google TTS quota management system initialized", level="INFO")
    else:
        log_message(log_widget, "Quota management system not available", level="WARNING")
    progress_data = load_progress(output_dir, SYNTHESIS_PROGRESS_FILE)
    processed_lines = 0
    if progress_data:
        processed_lines = progress_data.get('processed_lines', 0)
        log_message(log_widget, f"Resuming speech synthesis. Previously processed: {processed_lines} lines", level="INFO")
    lines_to_synthesize = []
    try:
        with open(input_text_file, "r", encoding="utf-8") as infile:
            lines_to_synthesize = infile.readlines()
    except Exception as e:
        log_message(log_widget, f"Error reading text file {input_text_file}: {e}", level="ERROR")
        status_callback(f"Error reading input file: {e}")
        stop_task_timer(task_name, output_dir)
        return
    total_lines = len(lines_to_synthesize)
    if total_lines == 0:
        log_message(log_widget, "Input text file is empty. Nothing to synthesize.", level="WARNING")
        status_callback("Input file is empty.")
        stop_task_timer(task_name, output_dir)
        return
    # Validate channels and check compatibility
    google_tts_client = None
    google_channels = []
    emotion_compatible_channels = []
    # สร้าง Google TTS client และตรวจสอบ channels
    for channel_id, config in channel_configs.items():
        engine_type = config.get('engine')
        if engine_type == "Google Cloud TTS":
            google_channels.append(channel_id)
            # ตรวจสอบความเข้ากันได้กับอารมณ์
            voice_name = config.get('name', '')
            thai_emotion_voices = ['th-TH-Neural2-C', 'th-TH-Standard-A']
            if any(supported_voice in voice_name for supported_voice in thai_emotion_voices):
                emotion_compatible_channels.append(channel_id)
    if google_channels and not GOOGLE_TTS_AVAILABLE:
        log_message(log_widget, "Google Cloud TTS selected but library not available.", level="ERROR")
        status_callback("Error: Google Cloud Text-to-Speech library not installed.")
        stop_task_timer(task_name, output_dir)
        return
    if google_channels:
        # สร้าง Google TTS client
        google_key_path_for_init = None
        for channel_id in google_channels:
            key_path = channel_configs[channel_id].get('google_key_path')
            if key_path and os.path.exists(key_path):
                google_key_path_for_init = key_path
                break
        if google_key_path_for_init:
            try:
                google_tts_client = texttospeech.TextToSpeechClient.from_service_account_json(google_key_path_for_init)
                log_message(log_widget, "Google Cloud TTS client initialized.", level="INFO")
            except Exception as e:
                log_message(log_widget, f"Failed to initialize Google Cloud TTS client: {e}", level="ERROR")
                status_callback(f"Error initializing Google TTS: {e}")
                stop_task_timer(task_name, output_dir)
                return
        else:
            log_message(log_widget, "Google Cloud TTS selected, but no valid key file found.", level="ERROR")
            status_callback("Error: No valid Google TTS key file found.")
            stop_task_timer(task_name, output_dir)
            return
    # Check gTTS availability
    gtts_channels = [cid for cid, config in channel_configs.items() if config.get('engine') == 'gTTS']
    if gtts_channels and not GTTS_AVAILABLE:
        log_message(log_widget, "gTTS selected but library not available.", level="ERROR")
        status_callback("Error: gTTS library not installed.")
        stop_task_timer(task_name, output_dir)
        return
    # Log compatibility information
    if use_auto_emotion:
        if emotion_compatible_channels:
            log_message(log_widget, f"Emotion analysis enabled for channels: {', '.join(emotion_compatible_channels)}", level="INFO")
        else:
            log_message(log_widget, "Emotion analysis enabled but no compatible channels found", level="WARNING")
    synthesis_errors = 0
    quota_errors = 0
    lines_to_process = lines_to_synthesize[processed_lines:] if processed_lines > 0 else lines_to_synthesize
    for i, line in enumerate(lines_to_process):
        current_line_index = processed_lines + i
        if stop_processing_flag.is_set():
            status_callback(f"Speech synthesis cancelled after {current_line_index + 1}/{total_lines} lines.")
            log_message(log_widget, "Speech synthesis cancelled by user.", level="INFO")
            stop_task_timer(task_name, output_dir)
            return
        progress = (current_line_index + 1) / total_lines
        progress_callback(progress)
        # Update time tracking
        total_time, session_time = update_task_timer(task_name, output_dir)
        estimated_remaining = estimate_completion_time(progress, total_time) if progress > 0 else 0
        time_status = f"Time: {format_time_duration(total_time)} | ETA: {format_time_duration(estimated_remaining)}"
        filename_base, _, _, channel_id, text_to_speak = parse_output_line(line)
        if not filename_base:
            filename_base = f"line_{current_line_index + 1}"
        if not text_to_speak or not text_to_speak.strip():
            log_message(log_widget, f"Skipping synthesis for line {current_line_index +1}: No text found.", level="WARNING")
            continue
        if not channel_id or not channel_id.startswith("Channel"):
            log_message(log_widget, f"Skipping synthesis for line {current_line_index + 1}: Invalid or missing Channel ID ('{channel_id}'). Expected format like 'Channel 1'.", level="WARNING")
            continue
        status_callback(f"Synthesizing line {current_line_index + 1}/{total_lines} using {channel_id}... | {time_status}")
        log_message(log_widget, f"Processing line {current_line_index + 1}: FileBase='{filename_base}', Channel='{channel_id}'", level="DEBUG")
        channel_config = channel_configs.get(channel_id)
        if not channel_config:
            log_message(log_widget, f"Configuration for {channel_id} not found. Skipping line {current_line_index + 1}.", level="WARNING")
            synthesis_errors += 1
            continue
        channel_engine = channel_config.get('engine')
        output_filename = f"{os.path.splitext(filename_base)[0]}_{channel_id.replace(' ','')}.{output_format}"
        output_path = os.path.join(output_dir, output_filename)
        try:
            text_bytes = calculate_text_bytes(text_to_speak)
            log_message(log_widget, f"Text size: {text_bytes} bytes", level="DEBUG")
            # ตรวจสอบความเข้ากันได้ของ emotion กับ channel
            use_emotion_for_channel = use_auto_emotion and (channel_id in emotion_compatible_channels)
            if text_bytes > 900:
                log_message(log_widget, f"Text exceeds 900 bytes ({text_bytes}), splitting into chunks with {regex_mode.name} mode...", level="INFO")
                text_chunks = split_text_by_sentences(text_to_speak, mode=regex_mode, custom_pattern=custom_regex, max_bytes=900)
                log_message(log_widget, f"Split into {len(text_chunks)} chunks", level="INFO")
                temp_audio_files = []
                for chunk_idx, chunk_text in enumerate(text_chunks):
                    chunk_filename = f"{os.path.splitext(filename_base)[0]}_{channel_id.replace(' ','')}_part{chunk_idx+1}.{output_format}"
                    chunk_output_path = os.path.join(output_dir, chunk_filename)
                    success = synthesize_single_chunk(
                        chunk_text,
                        chunk_output_path,
                        channel_engine,
                        channel_config,
                        output_format,
                        log_widget,
                        google_tts_client,
                        emotion_analyzer,
                        ssml_generator,
                        use_emotion_for_channel,
                        use_advanced_emotion,
                        quota_manager
                    )
                    if success:
                        temp_audio_files.append(chunk_output_path)
                    else:
                        log_message(log_widget, f"Failed to synthesize chunk {chunk_idx+1}", level="ERROR")
                        if "quota" in str(success).lower():
                            quota_errors += 1
                        else:
                            synthesis_errors += 1
                        break
                if len(temp_audio_files) == len(text_chunks):
                    log_message(log_widget, f"Merging {len(temp_audio_files)} audio chunks...", level="INFO")
                    merge_success = merge_audio_files(temp_audio_files, output_path)
                    if merge_success:
                        log_message(log_widget, f"Successfully merged audio for '{output_filename}'", level="INFO")
                    else:
                        log_message(log_widget, f"Failed to merge audio chunks for '{output_filename}'", level="ERROR")
                        synthesis_errors += 1
                else:
                    log_message(log_widget, f"Not all chunks synthesized successfully for '{filename_base}'", level="ERROR")
                    synthesis_errors += 1
                    for temp_file in temp_audio_files:
                        try:
                            os.remove(temp_file)
                        except:
                            pass
            else:
                success = synthesize_single_chunk(
                    text_to_speak,
                    output_path,
                    channel_engine,
                    channel_config,
                    output_format,
                    log_widget,
                    google_tts_client,
                    emotion_analyzer,
                    ssml_generator,
                    use_emotion_for_channel,
                    use_advanced_emotion,
                    quota_manager
                )
                if not success:
                    if "quota" in str(success).lower():
                        quota_errors += 1
                    else:
                        synthesis_errors += 1
            # Save progress
            progress_data = {
                'processed_lines': current_line_index + 1,
                'total_lines': total_lines,
                'channel_configs': channel_configs,
                'output_format': output_format,
                'use_auto_emotion': use_auto_emotion,
                'use_advanced_emotion': use_advanced_emotion,
                'synthesis_errors': synthesis_errors,
                'quota_errors': quota_errors,
                'last_updated': datetime.datetime.now().isoformat()
            }
            save_progress(output_dir, SYNTHESIS_PROGRESS_FILE, progress_data)
        except Exception as e:
            log_message(log_widget, f"Error synthesizing line {current_line_index + 1}: {e}", level="ERROR")
            if "quota" in str(e).lower():
                quota_errors += 1
            else:
                synthesis_errors += 1
    final_time = stop_task_timer(task_name, output_dir)
    # สร้างรายงานสรุป
    status_message = f"Speech synthesis complete. Processed {total_lines}/{total_lines} lines. Total time: {format_time_duration(final_time)}"
    if synthesis_errors > 0 or quota_errors > 0:
        error_details = []
        if synthesis_errors > 0:
            error_details.append(f"{synthesis_errors} synthesis errors")
        if quota_errors > 0:
            error_details.append(f"{quota_errors} quota errors")
        status_message += f" Encountered {', '.join(error_details)}."
        log_message(log_widget, f"Synthesis finished with errors: {', '.join(error_details)}", level="WARNING")
    else:
        log_message(log_widget, f"Synthesis finished successfully. Output files saved in {output_dir}", level="INFO")
    # แสดงสรุปการใช้งาน quota
    if quota_manager:
        try:
            quota_summary_text = quota_manager.get_full_usage_summary()
            log_message(log_widget, "\n" + quota_summary_text, level="INFO")
        except Exception as e:
            log_message(log_widget, f"Error generating quota summary: {e}", level="WARNING")
    status_callback(status_message)
    clear_progress(output_dir, SYNTHESIS_PROGRESS_FILE)
    if auto_shutdown:
        log_message(log_widget, "Auto shutdown requested. Shutting down Windows in 30 seconds...", level="INFO")
        shutdown_windows()
# ===============================================
# TEST VOICE FUNCTION
# ===============================================
def test_channel_voice(channel_config, output_dir, output_format, log_widget, emotion_analyzer=None, ssml_generator=None, use_auto_emotion=False, use_advanced_emotion=False):
    channel_engine = channel_config.get('engine')
    channel_id = channel_config.get('id', 'TestChannel')
    # ข้อความทดสอบที่หลากหลายสำหรับทดสอบอารมณ์
    if use_auto_emotion:
        test_text = f"ฉันรู้สึกดีใจมากที่ได้ทดสอบเสียงของ {channel_id} แต่ก็เป็นห่วงนิดหน่อยว่าจะออกมาดีไหม"
    else:
        test_text = f"This is a test voice synthesis for {channel_id}."
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"Test_{channel_id.replace(' ','')}_{timestamp}.{output_format}"
    output_path = os.path.join(output_dir, output_filename)
    try:
        # Clean up old test files
        for f in os.listdir(output_dir):
            if f.startswith(f"Test_{channel_id.replace(' ','')}_") and f.endswith(f".{output_format}"):
                os.remove(os.path.join(output_dir, f))
                log_message(log_widget, f"Removed previous test file: {f}", level="DEBUG")
    except Exception as e:
        log_message(log_widget, f"Could not clean up old test files: {e}", level="WARNING")
    log_message(log_widget, f"Starting test synthesis for {channel_id} -> {output_path}", level="INFO")
    try:
        # Initialize quota manager for testing
        quota_manager = create_quota_manager(output_dir)
        success = synthesize_single_chunk(
            test_text,
            output_path,
            channel_engine,
            channel_config,
            output_format,
            log_widget,
            None, # google_tts_client will be created inside if needed
            emotion_analyzer,
            ssml_generator,
            use_auto_emotion,
            use_advanced_emotion,
            quota_manager
        )
        if success:
            emotion_info = ""
            quota_info = ""
            # แสดงข้อมูลอารมณ์
            if use_auto_emotion and emotion_analyzer:
                try:
                    if use_advanced_emotion:
                        analysis = emotion_analyzer.analyze_advanced(test_text)
                        emotion_info = f"\nEmotion analysis (Advanced): "
                        if analysis.get('sentences'):
                            emotions = [s['emotion'] for s in analysis['sentences']]
                            emotion_info += f"{', '.join(emotions)}"
                    else:
                        analysis = emotion_analyzer.analyze_simple(test_text)
                        emotion_info = f"\nDetected emotion: {analysis['emotion']}"
                        if analysis.get('first_keyword'):
                            emotion_info += f" (keyword: {analysis['first_keyword']})"
                except Exception as e:
                    emotion_info = f"\nEmotion analysis error: {e}"
            # แสดงข้อมูล quota
            if quota_manager and channel_engine == "Google Cloud TTS":
                try:
                    key_path = channel_config.get('google_key_path')
                    voice_name = channel_config.get('name', '')
                    if key_path and voice_name:
                        voice_type = quota_manager.get_voice_type(voice_name)
                        quota_display = quota_manager.get_realtime_display_for_key(voice_type, key_path)
                        quota_info = f"\nQuota ({os.path.basename(key_path)}): {quota_display['formatted']}"
                except Exception as e:
                    quota_info = f"\nQuota info error: {e}"
            log_message(log_widget, f"Test voice saved to {output_path}{emotion_info}{quota_info}", level="INFO")
            messagebox.showinfo("Test Voice", f"Test audio saved as:\n{output_path}{emotion_info}{quota_info}")
        else:
            message = f"Test voice synthesis failed for {channel_id}"
            log_message(log_widget, message, level="ERROR")
            messagebox.showerror("Test Voice Error", message)
    except Exception as e:
        error_message = f"Error testing voice for {channel_id}: {e}"
        log_message(log_widget, error_message, level="ERROR")
        messagebox.showerror("Test Voice Error", error_message)
# ===============================================
# ===============================================
# MAIN APPLICATION CLASS
# ===============================================
class AudioProcessorApp(ctk.CTk):
   def __init__(self):
    super().__init__()
    self.title(f"{APP_NAME} v{APP_VERSION}")
    self.geometry(WINDOW_SIZE)
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme(THEME)
    self.option_add("*Font", DEFAULT_FONT)
    # Prevent window flashing
    self.wm_attributes('-topmost', False)
    self.focus_force = lambda: None
    # Initialize all variables first
    self._initialize_basic_variables()
    # Create widgets
    self._create_widgets()
    # Setup event handling
    self.protocol("WM_DELETE_WINDOW", self.on_closing)
    # Schedule post-initialization after UI is ready
    self.after(100, self._post_initialization)
   def _initialize_basic_variables(self):
    """Initialize all basic variables before creating widgets"""
    # Initialize variables
    self.current_task_thread = None
    self.input_folder = tk.StringVar()
    self.output_folder = tk.StringVar()
    self.google_stt_key_path = tk.StringVar()
    self.gemini_settings_path = tk.StringVar(value=DEFAULT_GEMINI_SETTINGS_FILE)
    self.copilot_key_path = tk.StringVar()
    self.chatgpt_key_path = tk.StringVar()
    self.google_tts_key_path = tk.StringVar()
    self.google_tts_voices = []
    self.google_tts_voice_file = tk.StringVar(value=DEFAULT_VOICE_FILE)
    self.tts_channel_widgets = {}
    self.tts_channel_configs = {}
    self.tts_next_channel_id = 1
    # Quota management variables
    self.quota_manager = None
    self.quota_widgets = {}
    # Auto shutdown variables
    self.auto_shutdown_freq = tk.BooleanVar()
    self.auto_shutdown_trans = tk.BooleanVar()
    self.auto_shutdown_translate = tk.BooleanVar()
    self.auto_shutdown_synthesis = tk.BooleanVar()
    # Emotion system variables
    self.use_auto_emotion_var = tk.BooleanVar(value=False)
    self.use_advanced_emotion_var = tk.BooleanVar(value=False)
    self.emotion_analyzer = None
    self.ssml_generator = None
    # Gender frequency range variables
    self.male_min_hz = tk.DoubleVar(value=DEFAULT_MALE_MIN_HZ)
    self.male_max_hz = tk.DoubleVar(value=DEFAULT_MALE_MAX_HZ)
    self.female_min_hz = tk.DoubleVar(value=DEFAULT_FEMALE_MIN_HZ)
    self.female_max_hz = tk.DoubleVar(value=DEFAULT_FEMALE_MAX_HZ)
    # Time tracking variables
    self.current_task_name = None
    self.time_update_timer = None
    # Auto-save timer
    self._auto_save_timer = None
    self._suppress_auto_save = False
   def _post_initialization(self):
    """Complete initialization after UI is ready"""
    try:
        # Load settings ตัวปิดเปิดออโต้โหลด
        #self.load_settings()
        # Initialize systems
        self.initialize_all_systems()
        # Start periodic tasks
        self.check_queue()
        self.update_time_display()
        self.after(5000, self.periodic_quota_update)
        # Show startup summary
        self.after(1000, self.show_startup_summary)
        # Validate UI state after initialization
        if not self.validate_ui_state():
            self.log_message_gui("UI state validation failed, attempting recovery", level="WARNING")
        # Check system health
        health_report = self.check_system_health()
        if not all(health_report.values()):
            self.log_message_gui("System health check found issues", level="WARNING")
        # Create enhanced emotion config if needed
        self.create_enhanced_emotion_config()
        self.log_message_gui("Application initialized successfully", level="INFO")
    except Exception as e:
        self.log_message_gui(f"Error during post-initialization: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
   def _create_widgets(self):
       self.grid_columnconfigure(0, weight=1)
       self.grid_rowconfigure(0, weight=1)
       self.grid_rowconfigure(1, weight=0)
       # Main tab view
       self.tab_view = ctk.CTkTabview(self)
       self.tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
       self.tab_view.add("Setup")
       self.tab_view.add("Frequency Analysis")
       self.tab_view.add("Transcription")
       self.tab_view.add("Translation")
       self.tab_view.add("Speech Synthesis")
       self.tab_view.add("Log")
       self._create_log_tab(self.tab_view.tab("Log"))       
       # Create tabs
       self._create_setup_tab(self.tab_view.tab("Setup"))
       self._create_frequency_tab(self.tab_view.tab("Frequency Analysis"))
       self._create_transcription_tab(self.tab_view.tab("Transcription"))
       self._create_translation_tab(self.tab_view.tab("Translation"))
       self._create_synthesis_tab(self.tab_view.tab("Speech Synthesis"))
       # Status frame
       self.status_frame = ctk.CTkFrame(self, height=30)
       self.status_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
       self.status_frame.grid_columnconfigure(0, weight=1)
       self.status_label = ctk.CTkLabel(self.status_frame, text="Idle", anchor="w")
       self.status_label.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
       self.progress_bar = ctk.CTkProgressBar(self.status_frame)
       self.progress_bar.grid(row=0, column=1, padx=10, pady=5, sticky="e")
       self.progress_bar.set(0)
       self.stop_button = ctk.CTkButton(self.status_frame, text="Stop Process", command=self.stop_current_task, state="disabled", width=100)
       self.stop_button.grid(row=0, column=2, padx=10, pady=5, sticky="e")
   def _create_setup_tab(self, tab):
       tab.grid_columnconfigure(1, weight=1)
               # === เพิ่มปุ่มใหม่ตรงนี้ ===
       project_management_frame = ctk.CTkFrame(tab)
       project_management_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
       load_project_button = ctk.CTkButton(project_management_frame, text="📂 โหลดโปรเจกต์ (Load Project)", command=self.load_project)
       load_project_button.pack(side="left", padx=10, pady=10)
       ctk.CTkLabel(project_management_frame, text="เลือกโฟลเดอร์โปรเจกต์ที่มีอยู่เพื่อโหลดการตั้งค่าทั้งหมด").pack(side="left", padx=10)
        # === สิ้นสุดส่วนที่เพิ่ม ===
       # Input folder
       ctk.CTkLabel(tab, text="Input Folder:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
       self.input_folder_entry = ctk.CTkEntry(tab, textvariable=self.input_folder, width=400)
       self.input_folder_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
       ctk.CTkButton(tab, text="Browse...", command=lambda: self.update_folder_path(self.input_folder, self.input_folder_entry, "Select Input Audio/Video Folder")).grid(row=1, column=2, padx=10, pady=10)
       # Output folder
       ctk.CTkLabel(tab, text="Output Folder:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
       self.output_folder_entry = ctk.CTkEntry(tab, textvariable=self.output_folder, width=400)
       self.output_folder_entry.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
       ctk.CTkButton(tab, text="Browse...", command=lambda: self.update_folder_path(self.output_folder, self.output_folder_entry, "Select Output Folder")).grid(row=2, column=2, padx=10, pady=10)
       # Progress management section
       progress_frame = ctk.CTkFrame(tab)
       progress_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
       ctk.CTkLabel(progress_frame, text="Progress Management:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10,5))
       button_frame = ctk.CTkFrame(progress_frame, fg_color="transparent")
       button_frame.pack(fill="x", padx=10, pady=(0,10))
       ctk.CTkButton(button_frame, text="Clear All Progress", 
                    command=self.clear_all_progress, 
                    fg_color="red", hover_color="darkred").pack(side="left", padx=(0,5))
       ctk.CTkButton(button_frame, text="View Progress Status", 
                    command=self.show_progress_status).pack(side="left", padx=5)
       # Description
       ctk.CTkLabel(tab, text="(คำอธิบาย)",
                    justify="left").grid(row=5, column=0, columnspan=3, padx=10, pady=20, sticky="w")
       # System management section
       system_frame = ctk.CTkFrame(tab)
       system_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
       ctk.CTkLabel(system_frame, text="System Management:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10,5))
       system_button_frame = ctk.CTkFrame(system_frame, fg_color="transparent")
       system_button_frame.pack(fill="x", padx=10, pady=(0,10))
       ctk.CTkButton(system_button_frame, text="Run System Tests", 
             command=self.run_system_tests,
             fg_color="blue", hover_color="darkblue").pack(side="left", padx=(0,5))
       ctk.CTkButton(system_button_frame, text="Validate System", 
             command=self.show_system_validation).pack(side="left", padx=5)
       ctk.CTkButton(system_button_frame, text="Create Backup", 
             command=self.create_system_backup,
             fg_color="green", hover_color="darkgreen").pack(side="left", padx=5)
       ctk.CTkButton(system_button_frame, text="Migrate Configs", 
             command=self.migrate_emotion_config,
             fg_color="orange", hover_color="darkorange").pack(side="left", padx=5)
       ctk.CTkButton(system_button_frame, text="Export Config", 
             command=self.export_system_configuration,
             fg_color="purple", hover_color="darkmagenta").pack(side="left", padx=5)
   def _create_frequency_tab(self, tab):
        tab.grid_columnconfigure(1, weight=1)
        # Header
        ctk.CTkLabel(tab, text="Analyze frequency to estimate gender (Male: ~50-90Hz, Female: ~91-500Hz) Support Audio only").grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        # Gender frequency range settings
        freq_settings_frame = ctk.CTkFrame(tab)
        freq_settings_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        freq_settings_frame.grid_columnconfigure(1, weight=1)
        freq_settings_frame.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(freq_settings_frame, text="Gender Frequency Ranges:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="w")
        # Male frequency range
        ctk.CTkLabel(freq_settings_frame, text="Male Hz:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        male_frame = ctk.CTkFrame(freq_settings_frame, fg_color="transparent")
        male_frame.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(male_frame, text="Min:").pack(side="left", padx=2)
        self.male_min_entry = ctk.CTkEntry(male_frame, textvariable=self.male_min_hz, width=60)
        self.male_min_entry.pack(side="left", padx=2)
        ctk.CTkLabel(male_frame, text="Max:").pack(side="left", padx=(10,2))
        self.male_max_entry = ctk.CTkEntry(male_frame, textvariable=self.male_max_hz, width=60)
        self.male_max_entry.pack(side="left", padx=2)
        # Female frequency range
        ctk.CTkLabel(freq_settings_frame, text="Female Hz:").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        female_frame = ctk.CTkFrame(freq_settings_frame, fg_color="transparent")
        female_frame.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(female_frame, text="Min:").pack(side="left", padx=2)
        self.female_min_entry = ctk.CTkEntry(female_frame, textvariable=self.female_min_hz, width=60)
        self.female_min_entry.pack(side="left", padx=2)
        ctk.CTkLabel(female_frame, text="Max:").pack(side="left", padx=(10,2))
        self.female_max_entry = ctk.CTkEntry(female_frame, textvariable=self.female_max_hz, width=60)
        self.female_max_entry.pack(side="left", padx=2)
        # Bind events for automatic adjustment
        self.male_max_hz.trace('w', self.on_male_max_change)
        self.female_min_hz.trace('w', self.on_female_min_change)
    # Save frequency settings button
        ctk.CTkButton(freq_settings_frame, text="Save Frequency Settings", 
                  command=self.save_frequency_settings).grid(row=2, column=0, columnspan=4, padx=5, pady=5)
        # Time range frame
        time_frame = ctk.CTkFrame(tab)
        time_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        time_frame.grid_columnconfigure(1, weight=0)
        time_frame.grid_columnconfigure(3, weight=0)
        ctk.CTkLabel(time_frame, text="Analyze Time Range (seconds):").pack(side="left", padx=5)
        ctk.CTkLabel(time_frame, text="Start:").pack(side="left", padx=(10, 0))
        self.freq_start_time_entry = ctk.CTkEntry(time_frame, width=60)
        self.freq_start_time_entry.pack(side="left", padx=5)
        self.freq_start_time_entry.insert(0, "0.0")
        ctk.CTkLabel(time_frame, text="End:").pack(side="left", padx=(10, 0))
        self.freq_end_time_entry = ctk.CTkEntry(time_frame, width=60)
        self.freq_end_time_entry.pack(side="left", padx=5)
        self.freq_end_time_entry.insert(0, "5.0")
        # Filter frame
        filter_frame = ctk.CTkFrame(tab)
        filter_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        self.filter_non_speech_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(filter_frame, text="Attempt to filter non-speech segments (basic)", variable=self.filter_non_speech_var).pack(side="left", padx=10)
        # Auto shutdown checkbox for frequency analysis
        ctk.CTkCheckBox(filter_frame, text="Auto shutdown Windows after completion", 
                   variable=self.auto_shutdown_freq).pack(side="right", padx=10)
        # Time tracking display frame
        self.freq_time_frame = ctk.CTkFrame(tab)
        self.freq_time_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        self.freq_time_label = ctk.CTkLabel(self.freq_time_frame, text="Time tracking will appear here during processing", 
                                       text_color="gray")
        self.freq_time_label.pack(padx=10, pady=5)
        # Start button
        ctk.CTkButton(tab, text="Start Frequency Analysis", command=self.start_frequency_analysis).grid(row=5, column=0, columnspan=3, padx=10, pady=20)
        # Description
        ctk.CTkLabel(tab, text="(คำอธิบาย)",
                justify="left").grid(row=6, column=0, columnspan=3, padx=10, pady=5, sticky="w")
   def on_male_max_change(self, *args):
        """When male max Hz changes, adjust female min Hz to match"""
        try:
            male_max = self.male_max_hz.get()
            self.female_min_hz.set(male_max + 1)
        except:
            pass
   def on_female_min_change(self, *args):
    """When female min Hz changes, adjust male max Hz to match"""
    try:
        female_min = self.female_min_hz.get()
        if female_min > 1:
            self.male_max_hz.set(female_min - 1)
    except:
        pass
   def save_frequency_settings(self):
    """Save frequency range settings"""
    try:
        male_min = self.male_min_hz.get()
        male_max = self.male_max_hz.get()
        female_min = self.female_min_hz.get()
        female_max = self.female_max_hz.get()
        if save_gender_frequency_settings(male_min, male_max, female_min, female_max):
            messagebox.showinfo("Settings Saved", "Gender frequency range settings saved successfully!")
            self.log_message_gui(f"Saved frequency settings: Male({male_min}-{male_max}Hz), Female({female_min}-{female_max}Hz)", level="INFO")
        else:
            messagebox.showerror("Error", "Failed to save frequency settings")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid frequency values: {e}")
   def _create_transcription_tab(self, tab):
    tab.grid_columnconfigure(1, weight=1)
    # Engine selection frame
    engine_frame = ctk.CTkFrame(tab)
    engine_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
    ctk.CTkLabel(engine_frame, text="Transcription Engine:").pack(side="left", padx=5)
    self.transcription_engine_var = tk.StringVar(value="Whisper" if WHISPER_AVAILABLE else ("Gemini" if GEMINI_AVAILABLE else ("Google Cloud STT" if GOOGLE_STT_AVAILABLE else "None")))
    engines = []
    if WHISPER_AVAILABLE: engines.append("Whisper")
    if GOOGLE_STT_AVAILABLE: engines.append("Google Cloud STT")
    if GEMINI_AVAILABLE: engines.append("Gemini")
    engines.extend(["Copilot (Not Implemented)", "ChatGPT (Not Implemented)", "Other (Not Implemented)"])
    if not engines: engines.append("None Available")
    self.transcription_engine_menu = ctk.CTkOptionMenu(engine_frame, variable=self.transcription_engine_var, values=engines, command=self.update_transcription_options)
    self.transcription_engine_menu.pack(side="left", padx=5)
    # Language selection
    ctk.CTkLabel(engine_frame, text="Language:").pack(side="left", padx=(20, 5))
    common_langs = ['auto', 'en', 'th', 'ja', 'ko', 'zh-cn', 'fr', 'de', 'es']
    all_langs = sorted(LANGUAGES.keys())
    lang_values = common_langs + [l for l in all_langs if l not in common_langs]
    self.transcription_lang_var = tk.StringVar(value="en")
    self.transcription_lang_combo = SearchableComboBox(engine_frame, variable=self.transcription_lang_var, values=lang_values, width=100)
    self.transcription_lang_combo.pack(side="left", padx=5)
    # Auto shutdown checkbox for transcription
    ctk.CTkCheckBox(engine_frame, text="Auto shutdown after completion", 
                  variable=self.auto_shutdown_trans).pack(side="right", padx=10)
    # Options frame (dynamic based on engine)
    self.transcription_options_frame = ctk.CTkFrame(tab, fg_color="transparent")
    self.transcription_options_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="nsew")
    self.transcription_options_frame.grid_columnconfigure(1, weight=1)
    # Prompt frame (for LLM engines)
    self.prompt_frame = ctk.CTkFrame(tab)
    self.prompt_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
    self.prompt_frame.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(self.prompt_frame, text="Optional Prompt for LLM (e.g., Gemini):").pack(side="top", anchor="w", padx=5)
    self.transcription_prompt_entry = ctk.CTkTextbox(self.prompt_frame, height=60, wrap="word")
    self.transcription_prompt_entry.pack(side="top", fill="x", expand=True, padx=5, pady=(0,5))
    self.transcription_prompt_entry.insert("1.0", "Accurately transcribe the entire audio file, line by line, into colloquial Thai.")
    self.prompt_frame.grid_remove()
    # เพิ่มใหม่: Time tracking display frame
    self.trans_time_frame = ctk.CTkFrame(tab)
    self.trans_time_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
    self.trans_time_label = ctk.CTkLabel(self.trans_time_frame, text="Time tracking will appear here during processing", 
                                      text_color="gray")
    self.trans_time_label.pack(padx=10, pady=5)
    # Start button
    ctk.CTkButton(tab, text="Start Transcription", command=self.start_transcription).grid(row=4, column=0, columnspan=3, padx=10, pady=20)
    # Status info
    ctk.CTkLabel(tab, text=f"Results (Filename|Gender|Age|Channel|Text) will be saved to: {OUTPUT_TRANSCRIPTION_FILE}").grid(row=5, column=0, columnspan=3, padx=10, pady=5, sticky="w")
    # Description
    ctk.CTkLabel(tab, text="(คำอธิบาย)",
                justify="left").grid(row=6, column=0, columnspan=3, padx=10, pady=5, sticky="w")
    # Initialize options based on default engine
    self.update_transcription_options(self.transcription_engine_var.get())
   def _create_translation_tab(self, tab):
        tab.grid_columnconfigure(1, weight=1)
        # Header info
        ctk.CTkLabel(tab, text=f"Translate text from: {OUTPUT_TRANSCRIPTION_FILE}").grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        ctk.CTkLabel(tab, text=f"Save translations to: {OUTPUT_TRANSLATION_FILE}").grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        # Language selection frame
        lang_frame = ctk.CTkFrame(tab)
        lang_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        ctk.CTkLabel(lang_frame, text="Target Language:").pack(side="left", padx=5)
        lang_codes = sorted(LANGUAGES.keys())
        lang_display = [f"{LANGUAGES[code].capitalize()} ({code})" for code in lang_codes]
        self.translation_target_lang_var = tk.StringVar(value="Thai (th)")
        self.lang_display_to_code = {f"{LANGUAGES[code].capitalize()} ({code})": code for code in lang_codes}
        self.translation_target_lang_combo = SearchableComboBox(lang_frame, variable=self.translation_target_lang_var, values=lang_display, width=200)
        self.translation_target_lang_combo.pack(side="left", padx=5)
        self.translation_target_lang_combo.set("Thai (th)")
        # Auto shutdown checkbox for translation
        ctk.CTkCheckBox(lang_frame, text="Auto shutdown after completion", 
                  variable=self.auto_shutdown_translate).pack(side="right", padx=10)
        # เพิ่มใหม่: Time tracking display frame
        self.translate_time_frame = ctk.CTkFrame(tab)
        self.translate_time_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        self.translate_time_label = ctk.CTkLabel(self.translate_time_frame, text="Time tracking will appear here during processing", 
                                          text_color="gray")
        self.translate_time_label.pack(padx=10, pady=5)
        # Start button
        ctk.CTkButton(tab, text="Start Translation", command=self.start_translation).grid(row=4, column=0, columnspan=3, padx=10, pady=20)
        # Description
        ctk.CTkLabel(tab, text="(คำอธิบาย)",
                justify="left").grid(row=5, column=0, columnspan=3, padx=10, pady=5, sticky="w")
   def _update_regex_description(self, selected_mode_name: str):
       """อัปเดตคำอธิบายและ UI ตามโหมด Regex ที่เลือก"""
       mode = RegexMode[selected_mode_name]
       # อัปเดตคำอธิบาย
       description = REGEX_DESCRIPTIONS.get(mode, "ไม่มีคำอธิบาย")
       self.regex_desc_label.configure(text=description)
       # แสดง/ซ่อน ช่องกรอก Custom Regex
       if mode == RegexMode.CUSTOM:
           self.custom_regex_entry.configure(state="normal")
       else:
           self.custom_regex_entry.configure(state="disabled")
   def _create_synthesis_tab(self, tab):
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_rowconfigure(6, weight=1)  # เปลี่ยนจาก 7 เป็น 6
    # Top controls - ทำให้ compact ขึ้น
    top_frame = ctk.CTkFrame(tab)
    top_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")  # ลด pady จาก 10 เป็น 5
    top_frame.grid_columnconfigure(1, weight=1)
    # Input file selection
    ctk.CTkLabel(top_frame, text="Input Text File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    self.tts_input_file_var = tk.StringVar(value="Select Input Translated_Sounds.txt")
    self.tts_input_file_entry = ctk.CTkEntry(top_frame, textvariable=self.tts_input_file_var, width=300)
    self.tts_input_file_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ctk.CTkButton(top_frame, text="Browse...", width=80, command=lambda: self.select_tts_input_file()).grid(row=0, column=2, padx=5, pady=5)
    # Output format และ auto shutdown ในบรรทัดเดียวกัน
    ctk.CTkLabel(top_frame, text="Output Format:").grid(row=0, column=3, padx=(20, 5), pady=5, sticky="w")
    self.tts_output_format_var = tk.StringVar(value="wav")
    formats = ["wav", "mp3", "ogg"]
    ctk.CTkOptionMenu(top_frame, variable=self.tts_output_format_var, values=formats, width=80).grid(row=0, column=4, padx=5, pady=5)
    # === ปุ่มใหม่สำหรับดูสรุป Quota ทั้งหมด ===
    ctk.CTkButton(top_frame, text="View Quota Summary", command=self.show_full_quota_summary).grid(row=0, column=5, padx=10, pady=5)
    ctk.CTkCheckBox(top_frame, text="Auto shutdown", variable=self.auto_shutdown_synthesis, width=150).grid(row=0, column=6, padx=10, pady=5)
    # Emotion controls frame - ทำให้ compact
    emotion_frame = ctk.CTkFrame(tab)
    emotion_frame.grid(row=1, column=0, padx=10, pady=3, sticky="ew")  # ลด pady
    # เพิ่มตัวแปรสำหรับ radio button (ตรวจสอบว่ามีอยู่แล้วหรือไม่)
    if not hasattr(self, 'emotion_mode_var'):
        self.emotion_mode_var = tk.StringVar(value="normal")
    # Radio buttons สำหรับเลือกโหมดอารมณ์ - แนวนอน
    radio_frame = ctk.CTkFrame(emotion_frame, fg_color="transparent")
    radio_frame.pack(fill="x", padx=5, pady=3)
    ctk.CTkRadioButton(radio_frame, text="สร้างเสียงปกติ", 
                      variable=self.emotion_mode_var, value="normal",
                      command=self.on_emotion_mode_change).pack(side="left", padx=5)
    ctk.CTkRadioButton(radio_frame, text="ใช้ระบบ SSML อัตโนมัติ (คำแรก)", 
                      variable=self.emotion_mode_var, value="auto_simple",
                      command=self.on_emotion_mode_change).pack(side="left", padx=5)
    ctk.CTkRadioButton(radio_frame, text="แบ่งวิเคราะห์ประโยคย่อย", 
                      variable=self.emotion_mode_var, value="auto_advanced",
                      command=self.on_emotion_mode_change).pack(side="left", padx=5)
    # ปุ่มจัดการในบรรทัดเดียวกัน
    ctk.CTkButton(radio_frame, text="จัดการคำศัพท์", width=120,
                command=self.open_emotion_manager).pack(side="right", padx=5)
    ctk.CTkButton(radio_frame, text="ทดสอบ", width=80,
                command=self.test_emotion_analysis).pack(side="right", padx=5)
    # ส่วนแสดงคำเตือนและข้อมูลเพิ่มเติม - ทำให้เล็กลง
    self.emotion_info_frame = ctk.CTkFrame(tab)
    self.emotion_info_frame.grid(row=2, column=0, padx=10, pady=2, sticky="ew")  # ลด pady
    self.emotion_info_label = ctk.CTkLabel(self.emotion_info_frame, text="", wraplength=900, justify="left", 
                                         font=ctk.CTkFont(size=13))  # ลดขนาดฟอนต์
    self.emotion_info_label.pack(padx=10, pady=3)  # ลด padding
    # Regex controls - ทำให้ compact
    regex_frame = ctk.CTkFrame(tab)
    regex_frame.grid(row=3, column=0, padx=10, pady=3, sticky="ew")
    regex_frame.grid_columnconfigure(2, weight=1)
    ctk.CTkLabel(regex_frame, text="Regex Mode:").grid(row=0, column=0, padx=5, pady=3, sticky="w")
    self.regex_mode_var = tk.StringVar(value=RegexMode.WHITESPACE.name)
    self.regex_mode_menu = ctk.CTkOptionMenu(
        regex_frame,
        variable=self.regex_mode_var,
        values=[mode.name for mode in RegexMode],
        command=self._update_regex_description,
        width=150
    )
    self.regex_mode_menu.grid(row=0, column=1, padx=5, pady=3, sticky="w")
    self.custom_regex_entry = ctk.CTkEntry(regex_frame, placeholder_text="Custom Regex Pattern", width=300)
    self.custom_regex_entry.grid(row=0, column=2, padx=5, pady=3, sticky="ew")
    # คำอธิบาย regex ในบรรทัดเดียว - ลดขนาด
    self.regex_desc_label = ctk.CTkLabel(regex_frame, text="", wraplength=800, justify="left", 
                                       font=ctk.CTkFont(size=12), text_color="gray")
    self.regex_desc_label.grid(row=1, column=0, columnspan=3, padx=10, pady=2, sticky="ew")
    # เรียกใช้ครั้งแรกเพื่อตั้งค่า UI ให้ถูกต้อง
    self._update_regex_description(self.regex_mode_var.get())
    # Channel management - ทำให้ compact
    channel_manage_frame = ctk.CTkFrame(tab)
    channel_manage_frame.grid(row=4, column=0, padx=10, pady=3, sticky="ew")
    channel_button_frame = ctk.CTkFrame(channel_manage_frame, fg_color="transparent")
    channel_button_frame.pack(fill="x", padx=5, pady=3)
    ctk.CTkButton(channel_button_frame, text="Add Synthesis Channel", command=self.add_tts_channel, width=150).pack(side="left", padx=5)
    ctk.CTkLabel(channel_button_frame, text="Configure voice channels based on '|Channel N|' markers", 
               font=ctk.CTkFont(size=13)).pack(side="left", padx=10)
    # Time tracking display frame - ทำให้เล็กลง
    self.synthesis_time_frame = ctk.CTkFrame(tab)
    self.synthesis_time_frame.grid(row=5, column=0, padx=10, pady=2, sticky="ew")
    self.synthesis_time_label = ctk.CTkLabel(self.synthesis_time_frame, text="Time tracking will appear here during processing", 
                                      text_color="gray", font=ctk.CTkFont(size=12))
    self.synthesis_time_label.pack(padx=10, pady=3)
    # Scrollable channels frame - ใช้พื้นที่ที่เหลือ
    self.channel_scroll_frame = ctk.CTkScrollableFrame(tab, label_text="Voice Channel Configurations")
    self.channel_scroll_frame.grid(row=6, column=0, padx=10, pady=5, sticky="nsew")
    self.channel_scroll_frame.grid_columnconfigure(0, weight=1)
    # Start button - ทำให้เล็กลง
    start_button_frame = ctk.CTkFrame(tab)
    start_button_frame.grid(row=7, column=0, padx=10, pady=5, sticky="ew")
    ctk.CTkButton(start_button_frame, text="Start Speech Synthesis", command=self.start_speech_synthesis, 
                height=35, font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
    # เรียกใช้ฟังก์ชันอัพเดทข้อมูลครั้งแรก
    self.on_emotion_mode_change()
   def _create_log_tab(self, tab):
       tab.grid_rowconfigure(0, weight=1)
       tab.grid_columnconfigure(0, weight=1)
       self.log_text = ctk.CTkTextbox(tab, state="disabled", wrap="word", font=(FONT_NAME, FONT_SIZE - 1))
       self.log_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
# ===============================================
# EMOTION SYSTEM METHODS
# ===============================================
   def on_auto_emotion_toggle(self):
    """เมื่อมีการเปิด/ปิดระบบ SSML อัตโนมัติ (เก่า - ใช้เพื่อ backward compatibility)"""
    # ฟังก์ชันนี้ยังคงไว้เพื่อไม่ให้ code เดิมเกิดข้อผิดพลาด
    # แต่การทำงานจริงจะใช้ on_emotion_mode_change แทน
    self.on_emotion_mode_change()
   def on_emotion_mode_change(self):
    """Handle emotion mode change with proper error handling and auto-save"""
    try:
        mode = getattr(self, 'emotion_mode_var', tk.StringVar(value='normal')).get()
        if mode == "normal":
            # Normal mode - no emotion analysis
            self.use_auto_emotion_var.set(False)
            self.use_advanced_emotion_var.set(False)
            self.emotion_analyzer = None
            self.ssml_generator = None
            info_text = "โหมดปกติ: สร้างเสียงตามการตั้งค่าของแต่ละ Channel โดยไม่วิเคราะห์อารมณ์"
            if hasattr(self, 'emotion_info_label'):
                self.emotion_info_label.configure(text=info_text, text_color="gray")
            self.log_message_gui("Emotion mode set to: Normal (manual settings)", level="INFO")
        elif mode == "auto_simple":
            # Simple auto mode
            self.use_auto_emotion_var.set(True)
            self.use_advanced_emotion_var.set(False)
            if self.initialize_emotion_system():
                compatible_voices = self.get_emotion_compatible_voices()
                info_text = f"โหมดอัตโนมัติ (คำแรกที่พบ): ใช้คำอารมณ์แรกที่พบในประโยคสร้างเสียงทั้งประโยค\n"
                info_text += f"⚠️ รองรับเฉพาะเสียงภาษาไทย: {', '.join(compatible_voices)}"
                if hasattr(self, 'emotion_info_label'):
                    self.emotion_info_label.configure(text=info_text, text_color="orange")
                self.log_message_gui("Emotion mode set to: Auto Simple (first keyword detection)", level="INFO")
            else:
                # Fallback to normal mode
                if hasattr(self, 'emotion_mode_var'):
                    self.emotion_mode_var.set("normal")
                self.on_emotion_mode_change()
                return
        elif mode == "auto_advanced":
            # Advanced auto mode
            self.use_auto_emotion_var.set(True)
            self.use_advanced_emotion_var.set(True)
            if self.initialize_emotion_system():
                compatible_voices = self.get_emotion_compatible_voices()
                info_text = f"โหมดขั้นสูง (แบ่งประโยคย่อย): วิเคราะห์อารมณ์แต่ละส่วนของประโยคแยกกัน\n"
                info_text += f"⚠️ รองรับเฉพาะเสียงภาษาไทย: {', '.join(compatible_voices)}"
                if hasattr(self, 'emotion_info_label'):
                    self.emotion_info_label.configure(text=info_text, text_color="blue")
                self.log_message_gui("Emotion mode set to: Auto Advanced (sentence splitting)", level="INFO")
            else:
                # Fallback to normal mode
                if hasattr(self, 'emotion_mode_var'):
                    self.emotion_mode_var.set("normal")
                self.on_emotion_mode_change()
                return
        # Auto-save after emotion mode change
        self.schedule_auto_save()
    except Exception as e:
        self.log_message_gui(f"Error in emotion mode change: {e}", level="ERROR")
        # Fallback to normal mode
        if hasattr(self, 'emotion_mode_var'):
            self.emotion_mode_var.set("normal")
   def get_emotion_compatible_voices(self):
    """ได้รับรายชื่อเสียงที่รองรับระบบอารมณ์"""
    # เสียงภาษาไทยที่รองรับ SSML และการควบคุมอารมณ์
    thai_emotion_voices = [
        'th-TH-Neural2-C',
        'th-TH-Standard-A'
    ]
    return thai_emotion_voices
   def initialize_quota_system(self):
    """เริ่มต้นระบบ quota management with enhanced error handling"""
    if self.output_folder.get():
        try:
            self.quota_manager = create_quota_manager(self.output_folder.get())
            if self.quota_manager:
                self.log_message_gui("Google TTS Quota system initialized", level="INFO")
                # Update quota displays for existing channels
                self.update_quota_displays()
                return True
            else:
                self.log_message_gui("Quota manager could not be created", level="WARNING")
        except Exception as e:
            self.log_message_gui(f"Failed to initialize quota system: {e}", level="ERROR")
    self.quota_manager = None
    return False
   def update_quota_displays(self):
    """อัพเดทการแสดงผล quota ทั้งหมด with error handling"""
    if not self.quota_manager:
        return
    try:
        for channel_id, widgets in self.quota_widgets.items():
            if channel_id in self.tts_channel_configs:
                config = self.tts_channel_configs[channel_id]
                voice_name = config.get('name', '')
                if voice_name:
                    voice_type = self.quota_manager.get_voice_type(voice_name)
                    quota_display = self.quota_manager.get_realtime_display(voice_type)
                    # อัพเดท label
                    if 'quota_label' in widgets:
                        color = "green"
                        if quota_display['percentage'] >= 90:
                            color = "red"
                        elif quota_display['percentage'] >= 70:
                            color = "orange"
                        widgets['quota_label'].configure(
                            text=quota_display['formatted'],
                            text_color=color
                        )
                    # อัพเดท progress bar
                    if 'quota_progress' in widgets:
                        progress = min(quota_display['percentage'] / 100.0, 1.0)
                        widgets['quota_progress'].set(progress)
                        # เปลี่ยนสีตามเปอร์เซ็นต์
                        if progress >= 0.9:
                            progress_color = "red"
                        elif progress >= 0.7:
                            progress_color = "orange"
                        else:
                            progress_color = "green"
                        widgets['quota_progress'].configure(progress_color=progress_color)
    except Exception as e:
        self.log_message_gui(f"Error updating quota displays: {e}", level="ERROR")
   def check_and_update_quota_for_synthesis(self, text, voice_name):
    """ตรวจสอบและอัพเดท quota สำหรับการสร้างเสียง"""
    if not self.quota_manager:
        return True, "Quota system not initialized"
    try:
        # ตรวจสอบ quota ก่อน
        can_proceed, quota_info = self.quota_manager.check_quota(text, voice_name)
        if not can_proceed:
            error_msg = handle_quota_exceeded_error(quota_info, voice_name, self.log_text)
            return False, error_msg
        # แสดงคำเตือนถ้าใกล้เต็ม
        warning = show_quota_warning(quota_info, voice_name)
        if warning:
            self.log_message_gui(warning, level="WARNING")
        return True, "Quota check passed"
    except Exception as e:
        self.log_message_gui(f"Error checking quota: {e}", level="ERROR")
        return True, f"Quota check error: {e}"
   def update_quota_after_synthesis(self, text, voice_name):
    """อัพเดท quota หลังสร้างเสียงสำเร็จ"""
    if not self.quota_manager:
        return
    try:
        usage_info = self.quota_manager.update_usage(text, voice_name)
        # อัพเดทการแสดงผล
        self.update_quota_displays()
        # Log การใช้งาน
        voice_type = usage_info.get('voice_type', 'unknown')
        usage_added = usage_info.get('usage_added', 0)
        total_usage = usage_info.get('total_usage', 0)
        unit = usage_info.get('unit', 'characters')
        self.log_message_gui(f"Quota updated - {voice_type}: +{usage_added:,} {unit}, total: {total_usage:,} {unit}", level="DEBUG")
    except Exception as e:
        self.log_message_gui(f"Error updating quota: {e}", level="ERROR")
   def create_quota_display_widgets(self, parent_frame, channel_id_str):
    """สร้าง widgets สำหรับแสดงผล quota - compact version"""
    # Current usage display - ทำให้เป็นแนวนอน
    usage_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
    usage_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=1)
    usage_frame.grid_columnconfigure(1, weight=1)
    ctk.CTkLabel(usage_frame, text="Quota:", font=ctk.CTkFont(size=13)).grid(row=0, column=0, padx=5, sticky="w")
    # Usage label - เล็กลง
    usage_label = ctk.CTkLabel(usage_frame, text="Loading...", text_color="blue", font=ctk.CTkFont(size=12))
    usage_label.grid(row=0, column=1, padx=5, sticky="w")
    # Progress bar - เล็กลง
    progress_bar = ctk.CTkProgressBar(usage_frame, width=150, height=12)
    progress_bar.grid(row=0, column=2, padx=5, sticky="w")
    progress_bar.set(0)
    # Refresh button - เล็กลง
    refresh_btn = ctk.CTkButton(usage_frame, text="🔄", width=25, height=20, font=ctk.CTkFont(size=12),
                               command=lambda: self.refresh_quota_display(channel_id_str))
    refresh_btn.grid(row=0, column=3, padx=2)
    # Store widgets
    if channel_id_str not in self.quota_widgets:
        self.quota_widgets[channel_id_str] = {}
    self.quota_widgets[channel_id_str].update({
        'quota_label': usage_label,
        'quota_progress': progress_bar,
        'refresh_btn': refresh_btn
    })
   def create_paid_feature_checkboxes(self, parent_frame, channel_id_str):
    """สร้าง checkboxes สำหรับ paid features - compact version"""
    # Get current voice to determine voice type
    voice_var = self.tts_channel_widgets[channel_id_str].get('google_voice_var')
    voice_name = voice_var.get() if voice_var else ''
    voice_type = 'standard'
    if self.quota_manager:
        voice_type = self.quota_manager.get_voice_type(voice_name)
    # Paid feature checkbox
    paid_var = tk.BooleanVar(value=False)
    # Dynamic text based on voice type - ย่อให้สั้นลง
    checkbox_texts = {
        'standard': 'Standard Paid ($4/1M chars)',
        'wavenet': 'WaveNet Paid ($16/1M chars)',
        'neural2': 'Neural2 Paid ($16/1M bytes)',
        'chirp': 'Chirp Paid ($16/1M chars)',
        'hd': 'HD Paid ($16/1M chars)',
        'studio': 'Studio Paid ($160/1M bytes)'
    }
    checkbox_text = checkbox_texts.get(voice_type, f'{voice_type.title()} Paid')
    # สร้าง frame สำหรับ checkbox
    paid_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
    paid_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=2)
    paid_checkbox = ctk.CTkCheckBox(
        paid_frame, 
        text=checkbox_text,
        variable=paid_var,
        command=lambda: self.on_paid_feature_toggle(channel_id_str, voice_type, paid_var.get()),
        font=ctk.CTkFont(size=13)
    )
    paid_checkbox.pack(side="left", padx=5)
    # Warning label - ย่อลง
    warning_label = ctk.CTkLabel(
        paid_frame, 
        text="⚠️ Enable when exceeding free quota",
        text_color="orange",
        font=ctk.CTkFont(size=11)
    )
    warning_label.pack(side="left", padx=10)
    # Store widgets
    self.quota_widgets[channel_id_str].update({
        'paid_checkbox': paid_checkbox,
        'paid_var': paid_var,
        'warning_label': warning_label,
        'voice_type': voice_type
    })
   def refresh_quota_display(self, channel_id_str):
    """รีเฟรช quota display สำหรับ channel"""
    if not self.quota_manager:
        return
    try:
        if channel_id_str in self.tts_channel_configs:
            config = self.tts_channel_configs[channel_id_str]
            voice_name = config.get('name', '')
            if voice_name:
                self.update_single_channel_quota_display(channel_id_str, voice_name)
                self.log_message_gui(f"Refreshed quota display for {channel_id_str}", level="DEBUG")
    except Exception as e:
        self.log_message_gui(f"Error refreshing quota display: {e}", level="ERROR")
    def update_single_channel_quota_display(self, channel_id_str, voice_name):
        """อัพเดท quota display สำหรับ channel เดียว โดยใช้ key ที่ถูกต้อง"""
        if not self.quota_manager or channel_id_str not in self.quota_widgets:
            return
        try:
            # === ดึง key_path จาก config ของ channel ===
            key_path = self.tts_channel_configs[channel_id_str].get('google_key_path')
            if not key_path:
                widgets = self.quota_widgets[channel_id_str]
                if 'quota_label' in widgets:
                    widgets['quota_label'].configure(text="No Key Assigned", text_color="red")
                if 'quota_progress' in widgets:
                    widgets['quota_progress'].set(0)
                return
            voice_type = self.quota_manager.get_voice_type(voice_name)
            # === เรียกใช้ฟังก์ชันใหม่ที่รับ key_path ===
            quota_display = self.quota_manager.get_realtime_display_for_key(voice_type, key_path)
            widgets = self.quota_widgets[channel_id_str]
            # อัพเดท label
            if 'quota_label' in widgets:
                color = "green"
                if quota_display['percentage'] >= 90:
                    color = "red"
                elif quota_display['percentage'] >= 70:
                    color = "orange"
                widgets['quota_label'].configure(
                    text=quota_display['formatted'],
                    text_color=color
                )
            # อัพเดท progress bar
            if 'quota_progress' in widgets:
                progress = min(quota_display['percentage'] / 100.0, 1.0)
                widgets['quota_progress'].set(progress)
                # เปลี่ยนสีตามเปอร์เซ็นต์
                if progress >= 0.9:
                    progress_color = "red"
                elif progress >= 0.7:
                    progress_color = "orange"
                else:
                    progress_color = "green"
                widgets['quota_progress'].configure(progress_color=progress_color)
            # อัพเดท paid feature checkbox
            if 'paid_var' in widgets:
                widgets['paid_var'].set(quota_display.get('paid_enabled', False))
        except Exception as e:
            self.log_message_gui(f"Error updating quota display for {channel_id_str}: {e}", level="ERROR")
   def update_paid_feature_checkbox(self, channel_id_str, voice_type):
    """อัพเดท paid feature checkbox ตามประเภทเสียง"""
    if channel_id_str not in self.quota_widgets:
        return
    widgets = self.quota_widgets[channel_id_str]
    checkbox_texts = {
        'standard': 'Standard Voice Paid Feature ($4/1M characters)',
        'wavenet': 'WaveNet Paid Feature ($16/1M characters)',
        'neural2': 'Neural2 Paid Feature ($16/1M bytes)',
        'chirp': 'Chirp Paid Feature ($16/1M characters)',
        'hd': 'HD Paid Feature ($16/1M characters)',
        'studio': 'Studio Paid Feature ($160/1M bytes)'
    }
    checkbox_text = checkbox_texts.get(voice_type, f'{voice_type.title()} Paid Feature')
    if 'paid_checkbox' in widgets:
        widgets['paid_checkbox'].configure(text=checkbox_text)
        widgets['voice_type'] = voice_type
        # รีเซ็ต checkbox state
        if 'paid_var' in widgets:
            current_state = False
            if self.quota_manager:
                current_state = self.quota_manager.paid_features.get(voice_type, False)
            widgets['paid_var'].set(current_state)
   def on_paid_feature_toggle(self, channel_id_str, voice_type, enabled):
        """เมื่อมีการเปลี่ยน paid feature checkbox with auto-save และ key-awareness"""
        if not self.quota_manager:
            return
        try:
            # === ดึง key_path จาก config ของ channel ===
            key_path = self.tts_channel_configs[channel_id_str].get('google_key_path')
            if not key_path:
                self.log_message_gui(f"Cannot toggle paid feature: No key assigned to {channel_id_str}", "ERROR")
                messagebox.showerror("Error", f"No Google Key assigned to {channel_id_str}.")
                # คืนค่า checkbox กลับไปเหมือนเดิม
                widgets = self.quota_widgets.get(channel_id_str)
                if widgets and 'paid_var' in widgets:
                    widgets['paid_var'].set(not enabled)
                return
            # === เรียกใช้ฟังก์ชันของ manager ที่รับ key_path ===
            self.quota_manager.enable_paid_feature(voice_type, key_path, enabled)
            status = "enabled" if enabled else "disabled"
            self.log_message_gui(f"Paid feature {status} for {voice_type} on key '{os.path.basename(key_path)}'", level="INFO")
            # Auto-save settings
            self.auto_save_settings()
            if enabled:
                messagebox.showinfo("Paid Feature Enabled",
                                    f"Paid feature enabled for {voice_type} voice on key:\n{os.path.basename(key_path)}\n\n"
                                    f"⚠️ Additional usage will incur charges on this specific key's billing account!")
        except Exception as e:
            self.log_message_gui(f"Error toggling paid feature: {e}", level="ERROR")
   def validate_emotion_selection(self):
    """ตรวจสอบการเลือกโหมดอารมณ์"""
    mode = self.emotion_mode_var.get()
    if mode in ["auto_simple", "auto_advanced"]:
        # ตรวจสอบว่ามี Channel ที่รองรับอารมณ์หรือไม่
        compatible_channels = []
        compatible_voices = self.get_emotion_compatible_voices()
        for channel_id, config in self.tts_channel_configs.items():
            voice_name = config.get('name', '')
            if any(compatible_voice in voice_name for compatible_voice in compatible_voices):
                compatible_channels.append(channel_id)
        if not compatible_channels:
            messagebox.showwarning("คำเตือน", 
                                 f"ไม่พบ Channel ที่รองรับระบบอารมณ์\n\n"
                                 f"เสียงที่รองรับ: {', '.join(compatible_voices)}\n\n"
                                 f"กรุณาเพิ่ม Channel ที่ใช้เสียงดังกล่าว หรือเปลี่ยนเป็นโหมดปกติ")
            return False
        else:
            self.log_message_gui(f"พบ Channel ที่รองรับอารมณ์: {', '.join(compatible_channels)}", level="INFO")
    return True
   def show_emotion_compatibility_warning(self):
    """แสดงคำเตือนเกี่ยวกับความเข้ากันได้ของเสียงกับอารมณ์"""
    mode = self.emotion_mode_var.get()
    if mode == "normal":
        return
    compatible_voices = self.get_emotion_compatible_voices()
    incompatible_channels = []
    for channel_id, config in self.tts_channel_configs.items():
        voice_name = config.get('name', '')
        if not any(compatible_voice in voice_name for compatible_voice in compatible_voices):
            incompatible_channels.append(f"{channel_id}: {voice_name}")
    if incompatible_channels:
        warning_text = f"คำเตือน: Channel ต่อไปนี้ไม่รองรับระบบอารมณ์และจะใช้การตั้งค่าปกติ:\n\n"
        warning_text += "\n".join(incompatible_channels)
        warning_text += f"\n\nเสียงที่รองรับระบบอารมณ์: {', '.join(compatible_voices)}"
        result = messagebox.showwarning("คำเตือนความเข้ากันได้", warning_text)
        # Log ข้อมูล
        self.log_message_gui(f"พบ Channel ที่ไม่รองรับอารมณ์: {len(incompatible_channels)} channels", level="WARNING")
   def validate_google_tts_quota_before_synthesis(self):
    """ตรวจสอบ quota ก่อนเริ่มสร้างเสียง"""
    if not self.quota_manager:
        return True, "Quota system not initialized"
    try:
        # อ่านไฟล์ input เพื่อประมาณการใช้งาน
        input_file = self.tts_input_file_var.get()
        if not os.path.exists(input_file):
            return False, "Input file not found"
        total_characters = 0
        total_bytes = 0
        voice_type_usage = {}
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            _, _, _, channel_id, text = parse_output_line(line)
            if channel_id and text and channel_id in self.tts_channel_configs:
                config = self.tts_channel_configs[channel_id]
                if config.get('engine') == 'Google Cloud TTS':
                    voice_name = config.get('name', '')
                    if voice_name:
                        voice_type = self.quota_manager.get_voice_type(voice_name)
                        usage = self.quota_manager.calculate_usage(text, voice_type)
                        if voice_type not in voice_type_usage:
                            voice_type_usage[voice_type] = 0
                        voice_type_usage[voice_type] += usage
        # ตรวจสอบว่าจะเกิด quota หรือไม่
        quota_warnings = []
        quota_errors = []
        for voice_type, estimated_usage in voice_type_usage.items():
            quota_display = self.quota_manager.get_realtime_display(voice_type)
            current_usage = quota_display['current_usage']
            limit = quota_display['limit']
            would_be_total = current_usage + estimated_usage
            if would_be_total > limit:
                paid_enabled = self.quota_manager.paid_features.get(voice_type, False)
                if not paid_enabled:
                    quota_errors.append(f"{voice_type}: {would_be_total:,}/{limit:,} {quota_display['unit']} (over by {would_be_total - limit:,})")
                else:
                    quota_warnings.append(f"{voice_type}: {would_be_total:,}/{limit:,} {quota_display['unit']} (paid feature enabled)")
            elif would_be_total > limit * 0.9:
                quota_warnings.append(f"{voice_type}: {would_be_total:,}/{limit:,} {quota_display['unit']} (>90%)")
        if quota_errors:
            error_msg = "Quota will be exceeded for:\n" + "\n".join(quota_errors)
            error_msg += "\n\nPlease enable paid features or reduce content."
            return False, error_msg
        if quota_warnings:
            warning_msg = "Quota warnings:\n" + "\n".join(quota_warnings)
            return True, warning_msg
        return True, "Quota check passed"
    except Exception as e:
        return True, f"Quota validation error: {e}"
   def show_pre_synthesis_quota_summary(self):
    """แสดงสรุป quota ก่อนเริ่มสร้างเสียง"""
    if not self.quota_manager:
        return
    try:
        quota_summary = self.quota_manager.get_usage_summary()
        summary_text = "=== Current Google TTS Quota Status ===\n\n"
        for voice_type, info in quota_summary.items():
            if info['current_usage'] > 0 or any(cid for cid, config in self.tts_channel_configs.items() 
                                               if config.get('engine') == 'Google Cloud TTS' 
                                               and self.quota_manager.get_voice_type(config.get('name', '')) == voice_type):
                summary_text += f"{info['name']}:\n"
                summary_text += f"  Current: {info['current_usage']:,}/{info['limit']:,} {info['unit']} ({info['percentage']:.1f}%)\n"
                summary_text += f"  Remaining: {info['remaining']:,} {info['unit']}\n"
                summary_text += f"  Paid Feature: {'✓' if info['paid_enabled'] else '✗'}\n\n"
        if summary_text.count('\n') > 3:  # มีข้อมูลจริง
            self.log_message_gui(summary_text.strip(), level="INFO")
    except Exception as e:
        self.log_message_gui(f"Error generating quota summary: {e}", level="ERROR")
   def periodic_quota_update(self):
    """อัพเดท quota display แบบ periodic"""
    try:
        if self.quota_manager and self.tts_channel_configs:
            self.update_quota_displays()
    except Exception as e:
        pass  # ไม่ต้อง log error ใน periodic update
    # Schedule next update
    self.after(5000, self.periodic_quota_update)  # อัพเดททุก 5 วินาที
   def open_emotion_manager(self):
    """Open emotion manager with enhanced error handling"""
    try:
        self.log_message_gui("Opening emotion manager...", level="DEBUG")
        # Initialize emotion analyzer if not exists
        if not hasattr(self, 'emotion_analyzer') or not self.emotion_analyzer:
            self.log_message_gui("Initializing emotion analyzer...", level="DEBUG")
            if not self.initialize_emotion_system():
                messagebox.showerror("Error", "Failed to initialize emotion system.\n\nPlease check the log for details.")
                return
        # Check if EmotionManagerWindow class exists
        try:
            emotion_manager = EmotionManagerWindow(self, self.emotion_analyzer)
            self.log_message_gui("EmotionManagerWindow created successfully", level="DEBUG")
            # Auto-save หลังปิด emotion manager window
            def on_emotion_manager_close():
                self.schedule_auto_save()
            emotion_manager.window.protocol("WM_DELETE_WINDOW", 
                lambda: [emotion_manager.window.destroy(), on_emotion_manager_close()])
        except NameError:
            messagebox.showerror("Error", "EmotionManagerWindow class not found.\n\nThis feature may not be available in this version.")
            self.log_message_gui("EmotionManagerWindow class not found", level="ERROR")
        except Exception as e:
            error_msg = f"Error creating EmotionManagerWindow: {e}"
            self.log_message_gui(error_msg, level="ERROR")
            messagebox.showerror("Error", f"Could not open emotion manager:\n{error_msg}")
    except Exception as e:
        error_msg = f"Error in open_emotion_manager: {e}"
        self.log_message_gui(error_msg, level="ERROR")
        messagebox.showerror("Error", f"Could not open emotion manager:\n{error_msg}")
   def test_emotion_analysis(self):
    """Test emotion analysis with enhanced error handling"""
    try:
        # Create test window
        test_window = ctk.CTkToplevel(self)
        test_window.title("ทดสอบการวิเคราะห์อารมณ์")
        test_window.geometry("600x400")
        test_window.transient(self)
        test_window.grab_set()
        # Input frame
        input_frame = ctk.CTkFrame(test_window)
        input_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(input_frame, text="ข้อความทดสอบ:").pack(anchor="w", padx=5)
        test_entry = ctk.CTkEntry(input_frame, width=500, placeholder_text="พิมพ์ข้อความเพื่อทดสอบการวิเคราะห์อารมณ์...")
        test_entry.pack(fill="x", padx=5, pady=5)
        # Mode selection
        mode_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        mode_frame.pack(fill="x", padx=5, pady=5)
        test_mode_var = tk.StringVar(value="simple")
        ctk.CTkRadioButton(mode_frame, text="แบบง่าย (Simple)", variable=test_mode_var, value="simple").pack(side="left", padx=5)
        ctk.CTkRadioButton(mode_frame, text="แบบขั้นสูง (Advanced)", variable=test_mode_var, value="advanced").pack(side="left", padx=5)
        # Result frame
        result_frame = ctk.CTkFrame(test_window)
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(result_frame, text="ผลการทดสอบ:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=5)
        result_text = ctk.CTkTextbox(result_frame)
        result_text.pack(fill="both", expand=True, padx=5, pady=5)
        result_text.configure(state="disabled")
        # Test function
        def run_test():
            test_text = test_entry.get().strip()
            if not test_text:
                messagebox.showwarning("แจ้งเตือน", "กรุณาพิมพ์ข้อความเพื่อทดสอบ")
                return
            try:
                # Initialize emotion system if needed
                if not hasattr(self, 'emotion_analyzer') or not self.emotion_analyzer:
                    if not self.initialize_emotion_system():
                        messagebox.showerror("Error", "Failed to initialize emotion system")
                        return
                mode = test_mode_var.get()
                # Perform analysis
                if mode == "simple":
                    analysis = self.emotion_analyzer.analyze_simple(test_text)
                    ssml = self.ssml_generator.create_simple_ssml(analysis) if self.ssml_generator else f"<speak>{test_text}</speak>"
                else:
                    analysis = self.emotion_analyzer.analyze_advanced(test_text)
                    ssml = self.ssml_generator.create_advanced_ssml(analysis) if self.ssml_generator else f"<speak>{test_text}</speak>"
                # Display results
                result_text.configure(state="normal")
                result_text.delete("1.0", tk.END)
                if mode == "simple":
                    result_text.insert("1.0", f"อารมณ์ที่ตรวจพบ: {analysis.get('emotion', 'unknown')}\n")
                    result_text.insert(tk.END, f"คะแนนความเชื่อมั่น: {analysis.get('confidence', 0)}\n")
                    if analysis.get('first_keyword'):
                        result_text.insert(tk.END, f"คำสำคัญที่พบ: {analysis['first_keyword']}\n")
                    result_text.insert(tk.END, f"คะแนนทั้งหมด: {analysis.get('all_scores', {})}\n\n")
                else:
                    result_text.insert("1.0", "โหมดขั้นสูง - วิเคราะห์แยกประโยค:\n")
                    if analysis.get('mode') == 'advanced' and analysis.get('sentences'):
                        for i, sentence_result in enumerate(analysis['sentences']):
                            emotion = sentence_result.get('emotion', 'unknown')
                            confidence = sentence_result.get('confidence', 0)
                            keyword = sentence_result.get('first_keyword', 'ไม่พบ')
                            result_text.insert(tk.END, f"ประโยคที่ {i+1}: {emotion} (คะแนน: {confidence}, คำสำคัญ: {keyword})\n")
                    result_text.insert(tk.END, "\n")
                result_text.insert(tk.END, "SSML ที่สร้าง:\n")
                result_text.insert(tk.END, ssml)
                result_text.configure(state="disabled")
                # Log การทดสอบ
                self.log_message_gui(f"Emotion test completed - Mode: {mode}, Detected: {analysis.get('emotion', 'unknown')}", level="INFO")
            except Exception as e:
                error_msg = f"เกิดข้อผิดพลาดในการทดสอบ:\n{e}"
                messagebox.showerror("ผิดพลาด", error_msg)
                self.log_message_gui(f"Emotion test failed: {e}", level="ERROR")
        # Test button
        ctk.CTkButton(input_frame, text="ทดสอบ", command=run_test).pack(pady=5)
        # Close button
        ctk.CTkButton(test_window, text="ปิด", command=test_window.destroy).pack(pady=10)
    except Exception as e:
        error_msg = f"Error creating test window: {e}"
        self.log_message_gui(error_msg, level="ERROR")
        messagebox.showerror("Error", error_msg)
   def migrate_emotion_config(self):
    """อัพเกรดไฟล์ emotion config เก่าให้รองรับฟีเจอร์ใหม่ with auto-save"""
    try:
       # ตรวจสอบว่ามีไฟล์ config เก่าหรือไม่
       config_file = os.path.join(self.output_folder.get() or ".", EMOTION_CONFIG_FILE)
       if os.path.exists(config_file):
           with open(config_file, 'r', encoding='utf-8') as f:
               old_config = json.load(f)
           # ตรวจสอบว่าต้อง migrate หรือไม่
           needs_migration = False
           for emotion, data in old_config.items():
               if isinstance(data, dict):
                   # ตรวจสอบว่ามี SSML config หรือไม่
                   if 'ssml' not in data:
                       needs_migration = True
                       break
                   # ตรวจสอบว่า keywords เป็น list หรือไม่
                   if 'keywords' in data and not isinstance(data['keywords'], list):
                       needs_migration = True
                       break
           if needs_migration:
               self.log_message_gui("Migrating emotion configuration to new format...", level="INFO")
               # สร้าง EmotionAnalyzer เพื่อให้ได้ default config
               emotion_analyzer = EmotionAnalyzer()
               migrated_config = emotion_analyzer.emotion_config.copy()
               # รวมข้อมูลเก่าเข้ากับข้อมูลใหม่
               for emotion, old_data in old_config.items():
                   if emotion in migrated_config:
                       if isinstance(old_data, dict) and 'keywords' in old_data:
                           # เก็บ keywords เดิม
                           old_keywords = old_data['keywords']
                           if isinstance(old_keywords, list):
                               migrated_config[emotion]['keywords'] = old_keywords
                           # เก็บ SSML settings เดิม (ถ้ามี)
                           if 'ssml' in old_data:
                               migrated_config[emotion]['ssml'] = old_data['ssml']
               # บันทึกไฟล์ที่ migrate แล้ว
               with open(config_file, 'w', encoding='utf-8') as f:
                   json.dump(migrated_config, f, ensure_ascii=False, indent=2)
               self.log_message_gui("Emotion configuration migrated successfully", level="INFO")
               # Auto-save หลัง migrate
               self.auto_save_settings()
               return True
       return False
    except Exception as e:
       self.log_message_gui(f"Error migrating emotion config: {e}", level="ERROR")
       return False
   def validate_system_integrity(self):
    """ตรวจสอบความสมบูรณ์ของระบบ"""
    issues = []
    try:
       # ตรวจสอบ emotion system
       if hasattr(self, 'emotion_analyzer') and self.emotion_analyzer:
           try:
               test_analysis = self.emotion_analyzer.analyze_simple("ทดสอบระบบ")
               if not test_analysis or 'emotion' not in test_analysis:
                   issues.append("Emotion analysis system not working properly")
           except Exception as e:
               issues.append(f"Emotion system error: {e}")
       # ตรวจสอบ quota system
       if hasattr(self, 'quota_manager') and self.quota_manager:
           try:
               test_quota = self.quota_manager.get_usage_summary()
               if not isinstance(test_quota, dict):
                   issues.append("Quota management system not working properly")
           except Exception as e:
               issues.append(f"Quota system error: {e}")
       # ตรวจสอบไฟล์ configuration ที่จำเป็น
       required_files = []
       output_dir = self.output_folder.get()
       if output_dir and os.path.exists(output_dir):
           settings_file = self.get_project_settings_file()
           if not os.path.exists(settings_file):
               required_files.append("app_settings.ini")
       if required_files:
           issues.append(f"Missing configuration files: {', '.join(required_files)}")
       # ตรวจสอบ TTS channels
       if not self.tts_channel_configs:
           issues.append("No TTS channels configured")
       else:
           for channel_id, config in self.tts_channel_configs.items():
               engine = config.get('engine')
               if engine == 'Google Cloud TTS':
                   key_path = config.get('google_key_path')
                   if not key_path or not os.path.exists(key_path):
                       issues.append(f"Google TTS key file missing for {channel_id}")
       # Log ผลการตรวจสอบ
       if issues:
           self.log_message_gui("System integrity check found issues:", level="WARNING")
           for issue in issues:
               self.log_message_gui(f"  - {issue}", level="WARNING")
       else:
           self.log_message_gui("System integrity check passed", level="INFO")
       return len(issues) == 0, issues
    except Exception as e:
       error_msg = f"Error during system validation: {e}"
       self.log_message_gui(error_msg, level="ERROR")
       return False, [error_msg]
   def run_system_tests(self):
    """รันการทดสอบระบบทั้งหมด with auto-save"""
    test_results = {
       'emotion_system': False,
       'quota_system': False,
       'config_system': False,
       'tts_system': False
    }
    self.log_message_gui("Starting comprehensive system tests...", level="INFO")
    try:
       # ทดสอบ emotion system
       try:
           if hasattr(self, 'emotion_analyzer') and self.emotion_analyzer:
               # ทดสอบ simple analysis
               simple_result = self.emotion_analyzer.analyze_simple("ฉันดีใจมาก")
               if simple_result and simple_result.get('emotion') == 'happy':
                   test_results['emotion_system'] = True
                   self.log_message_gui("✓ Emotion system test passed", level="INFO")
               else:
                   self.log_message_gui("✗ Emotion system test failed: incorrect analysis", level="ERROR")
           else:
               self.log_message_gui("✗ Emotion system test skipped: not initialized", level="WARNING")
       except Exception as e:
           self.log_message_gui(f"✗ Emotion system test failed: {e}", level="ERROR")
       # ทดสอบ quota system
       try:
           if hasattr(self, 'quota_manager') and self.quota_manager:
               # ทดสอบการคำนวณ usage
               test_usage = self.quota_manager.calculate_usage("test text", "standard")
               if test_usage > 0:
                   test_results['quota_system'] = True
                   self.log_message_gui("✓ Quota system test passed", level="INFO")
               else:
                   self.log_message_gui("✗ Quota system test failed: usage calculation error", level="ERROR")
           else:
               self.log_message_gui("✗ Quota system test skipped: not initialized", level="WARNING")
       except Exception as e:
           self.log_message_gui(f"✗ Quota system test failed: {e}", level="ERROR")
       # ทดสอบ config system
       try:
           # ทดสอบการบันทึกและโหลด settings
           test_config = configparser.ConfigParser()
           test_config['TestSection'] = {'test_key': 'test_value'}
           test_config_path = os.path.join(self.output_folder.get() or ".", "test_config.ini")
           with open(test_config_path, 'w', encoding='utf-8') as f:
               test_config.write(f)
           # ทดสอบการอ่านกลับ
           read_config = configparser.ConfigParser()
           read_config.read(test_config_path, encoding='utf-8')
           if read_config.get('TestSection', 'test_key') == 'test_value':
               test_results['config_system'] = True
               self.log_message_gui("✓ Configuration system test passed", level="INFO")
           else:
               self.log_message_gui("✗ Configuration system test failed: read/write error", level="ERROR")
           # ลบไฟล์ทดสอบ
           if os.path.exists(test_config_path):
               os.remove(test_config_path)
       except Exception as e:
           self.log_message_gui(f"✗ Configuration system test failed: {e}", level="ERROR")
       # ทดสอบ TTS system
       try:
           if self.tts_channel_configs:
               # ตรวจสอบว่ามี Google TTS channels พร้อมใช้งาน
               google_channels = [cid for cid, config in self.tts_channel_configs.items() 
                                if config.get('engine') == 'Google Cloud TTS']
               if google_channels:
                   # ตรวจสอบ key file อย่างน้อย 1 channel
                   for channel_id in google_channels:
                       config = self.tts_channel_configs[channel_id]
                       key_path = config.get('google_key_path')
                       if key_path and os.path.exists(key_path):
                           test_results['tts_system'] = True
                           break
                   if test_results['tts_system']:
                       self.log_message_gui("✓ TTS system test passed", level="INFO")
                   else:
                       self.log_message_gui("✗ TTS system test failed: no valid key files", level="ERROR")
               else:
                   test_results['tts_system'] = True  # ถ้าไม่มี Google TTS ก็ถือว่าผ่าน
                   self.log_message_gui("✓ TTS system test passed (no Google TTS channels)", level="INFO")
           else:
               self.log_message_gui("✗ TTS system test failed: no channels configured", level="ERROR")
       except Exception as e:
           self.log_message_gui(f"✗ TTS system test failed: {e}", level="ERROR")
       # สรุปผลการทดสอบ
       passed_tests = sum(test_results.values())
       total_tests = len(test_results)
       self.log_message_gui(f"System tests completed: {passed_tests}/{total_tests} passed", level="INFO")
       # Auto-save หลังทดสอบ
       self.auto_save_settings()
       if passed_tests == total_tests:
           self.log_message_gui("🎉 All systems are working correctly!", level="INFO")
           return True
       else:
           failed_systems = [system for system, passed in test_results.items() if not passed]
           self.log_message_gui(f"⚠️ Failed systems: {', '.join(failed_systems)}", level="WARNING")
           return False
    except Exception as e:
       self.log_message_gui(f"Critical error during system tests: {e}", level="ERROR")
       return False
   def create_system_backup(self):
    """สร้าง backup ของการตั้งค่าระบบทั้งหมด with auto-save"""
    try:
       output_dir = self.output_folder.get()
       if not output_dir or not os.path.exists(output_dir):
           messagebox.showerror("Error", "Please set output folder before creating backup")
           return False
       # สร้างโฟลเดอร์ backup
       backup_dir = os.path.join(output_dir, "system_backup")
       timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
       timestamped_backup_dir = os.path.join(backup_dir, f"backup_{timestamp}")
       os.makedirs(timestamped_backup_dir, exist_ok=True)
       # รายการไฟล์ที่ต้อง backup
       files_to_backup = [
           SETTINGS_FILE,
           EMOTION_CONFIG_FILE,
           GOOGLE_TTS_QUOTA_FILE,
           TIME_TRACKING_FILE
       ]
       backed_up_files = []
       for filename in files_to_backup:
           source_path = os.path.join(output_dir, filename)
           if os.path.exists(source_path):
               dest_path = os.path.join(timestamped_backup_dir, filename)
               shutil.copy2(source_path, dest_path)
               backed_up_files.append(filename)
       # สร้างไฟล์ข้อมูล backup
       backup_info = {
           'timestamp': timestamp,
           'backed_up_files': backed_up_files,
           'app_version': APP_VERSION,
           'system_info': {
               'emotion_system_enabled': hasattr(self, 'emotion_analyzer') and self.emotion_analyzer is not None,
               'quota_system_enabled': hasattr(self, 'quota_manager') and self.quota_manager is not None,
               'total_channels': len(self.tts_channel_configs),
               'emotion_mode': getattr(self, 'emotion_mode_var', tk.StringVar()).get()
           }
        }
       backup_info_path = os.path.join(timestamped_backup_dir, "backup_info.json")
       with open(backup_info_path, 'w', encoding='utf-8') as f:
           json.dump(backup_info, f, ensure_ascii=False, indent=2)
       self.log_message_gui(f"System backup created: {timestamped_backup_dir}", level="INFO")
       self.log_message_gui(f"Backed up files: {', '.join(backed_up_files)}", level="INFO")
       # Auto-save หลัง backup
       self.auto_save_settings()
       messagebox.showinfo("Backup Complete", 
                         f"System backup created successfully!\n\n"
                         f"Location: {timestamped_backup_dir}\n"
                         f"Files backed up: {len(backed_up_files)}")
       return True
    except Exception as e:
       error_msg = f"Error creating system backup: {e}"
       self.log_message_gui(error_msg, level="ERROR")
       messagebox.showerror("Backup Error", error_msg)
       return False
   def export_system_configuration(self):
    """ส่งออกการตั้งค่าระบบทั้งหมด with auto-save"""
    try:
       output_dir = self.output_folder.get()
       if not output_dir or not os.path.exists(output_dir):
           messagebox.showerror("Error", "Please set output folder before exporting configuration")
           return False
       # สร้างไฟล์ export
       timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
       export_filename = f"system_config_export_{timestamp}.json"
       export_path = os.path.join(output_dir, export_filename)
       # รวบรวมข้อมูลการตั้งค่าทั้งหมด
       export_data = {
           'metadata': {
               'app_name': APP_NAME,
               'app_version': APP_VERSION,
               'export_timestamp': timestamp,
               'export_date': datetime.datetime.now().isoformat()
           },
           'paths': {
               'input_folder': self.input_folder.get(),
               'output_folder': self.output_folder.get(),
               'google_voice_file': self.google_tts_voice_file.get()
           },
           'emotion_settings': {
               'mode': getattr(self, 'emotion_mode_var', tk.StringVar()).get(),
               'use_auto_emotion': self.use_auto_emotion_var.get(),
               'use_advanced_emotion': self.use_advanced_emotion_var.get()
           },
           'gender_frequency': {
               'male_min_hz': self.male_min_hz.get(),
               'male_max_hz': self.male_max_hz.get(),
               'female_min_hz': self.female_min_hz.get(),
               'female_max_hz': self.female_max_hz.get()
           },
           'tts_channels': self.tts_channel_configs.copy(),
           'system_status': {
               'emotion_system_available': hasattr(self, 'emotion_analyzer') and self.emotion_analyzer is not None,
               'quota_system_available': hasattr(self, 'quota_manager') and self.quota_manager is not None,
               'whisper_available': WHISPER_AVAILABLE,
               'google_stt_available': GOOGLE_STT_AVAILABLE,
               'gemini_available': GEMINI_AVAILABLE,
               'google_tts_available': GOOGLE_TTS_AVAILABLE,
               'gtts_available': GTTS_AVAILABLE
           }
        }
       # เพิ่มข้อมูล quota ถ้ามี
       if hasattr(self, 'quota_manager') and self.quota_manager:
           quota_summary = self.quota_manager.get_usage_summary()
           export_data['quota_status'] = quota_summary
           export_data['paid_features'] = self.quota_manager.paid_features.copy()
       # บันทึกไฟล์
       with open(export_path, 'w', encoding='utf-8') as f:
           json.dump(export_data, f, ensure_ascii=False, indent=2)
       self.log_message_gui(f"System configuration exported: {export_path}", level="INFO")
       # Auto-save หลัง export
       self.auto_save_settings()
       messagebox.showinfo("Export Complete", 
                         f"System configuration exported successfully!\n\n"
                         f"File: {export_filename}\n"
                         f"Location: {output_dir}")
       return True
    except Exception as e:
       error_msg = f"Error exporting system configuration: {e}"
       self.log_message_gui(error_msg, level="ERROR")
       messagebox.showerror("Export Error", error_msg)
       return False
   def show_system_validation(self):
    """แสดงผลการตรวจสอบระบบ"""
    is_valid, issues = self.validate_system_integrity()
    if is_valid:
       messagebox.showinfo("System Validation", 
                         "✅ System validation passed!\n\n"
                         "All components are working correctly.")
    else:
       issue_text = "\n".join([f"• {issue}" for issue in issues])
       messagebox.showwarning("System Validation", 
                            f"⚠️ System validation found issues:\n\n{issue_text}\n\n"
                            f"Please check the log for more details.")
   def initialize_all_systems(self):
    """เริ่มต้นระบบทั้งหมด with auto-save"""
    try:
       # Migrate configs if needed
       self.migrate_emotion_config()
       # Initialize emotion system
       if hasattr(self, 'emotion_mode_var'):
           mode = self.emotion_mode_var.get()
           if mode in ['auto_simple', 'auto_advanced']:
               self.on_emotion_mode_change()
       # Initialize quota system
       if self.output_folder.get():
           self.initialize_quota_system()
       # Validate system
       is_valid, issues = self.validate_system_integrity()
       # Auto-save after initialization
       self.auto_save_settings()
       if not is_valid:
           self.log_message_gui(f"System initialization completed with {len(issues)} issues", level="WARNING")
       else:
           self.log_message_gui("All systems initialized successfully", level="INFO")
       return is_valid
    except Exception as e:
       self.log_message_gui(f"Error during system initialization: {e}", level="ERROR")
       return False
   def on_application_startup(self):
    """เมื่อแอปพลิเคชันเริ่มต้น with comprehensive initialization"""
    try:
       # Load settings first
       self.load_settings()
       # Initialize all systems
       self.initialize_all_systems()
       # Show startup summary
       self.show_startup_summary()
    except Exception as e:
      self.log_message_gui(f"Error during application startup: {e}", level="ERROR")
   def show_startup_summary(self):
    """แสดงสรุปการเริ่มต้นระบบ"""
    try:
      summary_lines = []
      summary_lines.append(f"=== {APP_NAME} v{APP_VERSION} ===")
      summary_lines.append("")
      # System status
      emotion_status = "✓" if hasattr(self, 'emotion_analyzer') and self.emotion_analyzer else "✗"
      quota_status = "✓" if hasattr(self, 'quota_manager') and self.quota_manager else "✗"
      summary_lines.append("System Status:")
      summary_lines.append(f"  Emotion Analysis: {emotion_status}")
      summary_lines.append(f"  Quota Management: {quota_status}")
      summary_lines.append(f"  TTS Channels: {len(self.tts_channel_configs)}")
      summary_lines.append("")
      # Available engines
      engines = []
      if WHISPER_AVAILABLE:
          engines.append("Whisper")
      if GOOGLE_STT_AVAILABLE:
          engines.append("Google STT")
      if GEMINI_AVAILABLE:
          engines.append("Gemini")
      if GOOGLE_TTS_AVAILABLE:
          engines.append("Google TTS")
      if GTTS_AVAILABLE:
          engines.append("gTTS")
      summary_lines.append(f"Available Engines: {', '.join(engines) if engines else 'None'}")
      summary_lines.append("")
      # Paths
      if self.input_folder.get():
          summary_lines.append(f"Input Folder: {self.input_folder.get()}")
      if self.output_folder.get():
          summary_lines.append(f"Output Folder: {self.output_folder.get()}")
      summary_text = "\n".join(summary_lines)
      self.log_message_gui(summary_text, level="INFO")
    except Exception as e:
      self.log_message_gui(f"Error generating startup summary: {e}", level="WARNING")
# ===============================================
# SETTINGS MANAGEMENT
# ===============================================
   def load_project(self):
        """Opens a dialog to select a project folder and loads its settings."""
        project_folder = filedialog.askdirectory(title="เลือกโฟลเดอร์โปรเจกต์ (Select Project Folder)")
        if not project_folder:
            self.log_message_gui("Project loading cancelled.", level="INFO")
            return
        # ตั้งค่าทั้ง Input และ Output ไปที่โฟลเดอร์โปรเจกต์เป็นค่าเริ่มต้น
        self.output_folder.set(project_folder)
        self.input_folder.set(project_folder) 
        # อัปเดต UI ให้แสดง path ที่เลือก
        self.output_folder_entry.delete(0, tk.END)
        self.output_folder_entry.insert(0, project_folder)
        self.input_folder_entry.delete(0, tk.END)
        self.input_folder_entry.insert(0, project_folder)
        self.log_message_gui(f"Attempting to load project from: {project_folder}", level="INFO")
        # โหลดไฟล์ app_settings.ini จากโฟลเดอร์ที่เลือก
        try:
            self.load_settings()
            # หลังจากโหลดค่าทั้งหมดแล้ว ให้เริ่มต้นระบบที่เกี่ยวข้องใหม่
            self.initialize_all_systems()
            messagebox.showinfo("Project Loaded", f"โปรเจกต์ '{os.path.basename(project_folder)}' ถูกโหลดเรียบร้อยแล้ว")
        except Exception as e:
            self.log_message_gui(f"Failed to load project settings: {e}", level="ERROR")
            messagebox.showerror("Load Error", f"ไม่สามารถโหลดการตั้งค่าของโปรเจกต์ได้:\n{e}")
   def load_settings(self):
    """Load folder paths and other settings from the project-specific settings file with enhanced error handling"""
    config = configparser.ConfigParser()
    # ใช้ project-specific settings file
    settings_file = self.get_project_settings_file()
    self.log_message_gui(f"Loading settings from: {settings_file}", level="DEBUG")
    if os.path.exists(settings_file):
        try:
            config.read(settings_file, encoding='utf-8')
            # Load Paths section
            if 'Paths' in config:
                input_path = config.get('Paths', 'input_folder', fallback='')
                output_path = config.get('Paths', 'output_folder', fallback='')
                voice_file_path = config.get('Paths', 'google_voice_file', fallback=DEFAULT_VOICE_FILE)
                # Set paths
                self.input_folder.set(input_path)
                self.output_folder.set(output_path)
                self.google_tts_voice_file.set(voice_file_path)
                # Load voices if file exists
                if voice_file_path != DEFAULT_VOICE_FILE and os.path.exists(voice_file_path):
                    self.load_google_voices()
            # Load Emotion settings with error handling
            if 'Emotion' in config:
                try:
                    emotion_mode = config.get('Emotion', 'emotion_mode', fallback='normal')
                    if hasattr(self, 'emotion_mode_var'):
                        self.emotion_mode_var.set(emotion_mode)
                    else:
                        # Create emotion_mode_var if not exists
                        self.emotion_mode_var = tk.StringVar(value=emotion_mode)
                    # Backward compatibility
                    use_auto = config.getboolean('Emotion', 'use_auto_emotion', fallback=False)
                    use_advanced = config.getboolean('Emotion', 'use_advanced_emotion', fallback=False)
                    self.use_auto_emotion_var.set(use_auto)
                    self.use_advanced_emotion_var.set(use_advanced)
                except Exception as e:
                    self.log_message_gui(f"Error loading emotion settings: {e}", level="WARNING")
            # Load Gender Frequency settings
            if 'GenderFrequency' in config:
                try:
                    male_min = config.getfloat('GenderFrequency', 'male_min_hz', fallback=DEFAULT_MALE_MIN_HZ)
                    male_max = config.getfloat('GenderFrequency', 'male_max_hz', fallback=DEFAULT_MALE_MAX_HZ)
                    female_min = config.getfloat('GenderFrequency', 'female_min_hz', fallback=DEFAULT_FEMALE_MIN_HZ)
                    female_max = config.getfloat('GenderFrequency', 'female_max_hz', fallback=DEFAULT_FEMALE_MAX_HZ)
                    self.male_min_hz.set(male_min)
                    self.male_max_hz.set(male_max)
                    self.female_min_hz.set(female_min)
                    self.female_max_hz.set(female_max)
                except Exception as e:
                    self.log_message_gui(f"Error loading frequency settings: {e}", level="WARNING")
            # Load TTS channel configurations
            if 'TTSChannels' in config:
                try:
                    # Suppress auto-save during loading
                    self._suppress_auto_save = True
                    channels_json = config.get('TTSChannels', 'channel_configs', fallback='{}')
                    saved_channels = json.loads(channels_json)
                    # Clear existing channels first
                    self._clear_all_tts_channels()
                    # Load saved channels
                    for channel_id, channel_config in saved_channels.items():
                        self._add_tts_channel_from_config(channel_config, channel_id)
                    self.log_message_gui(f"Loaded {len(saved_channels)} TTS channels", level="INFO")
                except json.JSONDecodeError as e:
                    self.log_message_gui(f"Invalid TTS channel configuration JSON: {e}", level="ERROR")
                except Exception as e:
                    self.log_message_gui(f"Error loading TTS channels: {e}", level="ERROR")
                finally:
                    self._suppress_auto_save = False
            # Load Auto shutdown settings
            if 'AutoShutdown' in config:
                try:
                    self.auto_shutdown_freq.set(config.getboolean('AutoShutdown', 'frequency', fallback=False))
                    self.auto_shutdown_trans.set(config.getboolean('AutoShutdown', 'transcription', fallback=False))
                    self.auto_shutdown_translate.set(config.getboolean('AutoShutdown', 'translation', fallback=False))
                    self.auto_shutdown_synthesis.set(config.getboolean('AutoShutdown', 'synthesis', fallback=False))
                except Exception as e:
                    self.log_message_gui(f"Error loading auto shutdown settings: {e}", level="WARNING")
            # Load API Keys paths
            if 'APIKeys' in config:
                try:
                    google_stt_path = config.get('APIKeys', 'google_stt_key', fallback='')
                    gemini_settings_path = config.get('APIKeys', 'gemini_settings', fallback=DEFAULT_GEMINI_SETTINGS_FILE)
                    google_tts_path = config.get('APIKeys', 'google_tts_key', fallback='')
                    self.google_stt_key_path.set(google_stt_path)
                    self.gemini_settings_path.set(gemini_settings_path)
                    self.google_tts_key_path.set(google_tts_path)
                except Exception as e:
                    self.log_message_gui(f"Error loading API key paths: {e}", level="WARNING")
            # Load UI States
            if 'UIStates' in config:
                try:
                    # Transcription engine
                    trans_engine = config.get('UIStates', 'transcription_engine', fallback='Whisper' if WHISPER_AVAILABLE else 'None')
                    if hasattr(self, 'transcription_engine_var'):
                        self.transcription_engine_var.set(trans_engine)
                    # Translation language
                    trans_lang = config.get('UIStates', 'translation_language', fallback='Thai (th)')
                    if hasattr(self, 'translation_target_lang_var'):
                        self.translation_target_lang_var.set(trans_lang)
                    # Output format
                    output_format = config.get('UIStates', 'output_format', fallback='wav')
                    if hasattr(self, 'tts_output_format_var'):
                        self.tts_output_format_var.set(output_format)
                    # Input file for TTS
                    tts_input = config.get('UIStates', 'tts_input_file', fallback='Select Input Translated_Sounds.txt')
                    if hasattr(self, 'tts_input_file_var'):
                        self.tts_input_file_var.set(tts_input)
                except Exception as e:
                    self.log_message_gui(f"Error loading UI states: {e}", level="WARNING")
            self.log_message_gui(f"Settings loaded successfully from: {settings_file}", level="INFO")
        except configparser.Error as e:
            self.log_message_gui(f"Warning: Could not read settings file: {e}", level="WARNING")
        except Exception as e:
            self.log_message_gui(f"Error loading settings: {e}", level="ERROR")
    else:
        self.log_message_gui(f"No project settings found. Will create new project settings.", level="INFO")
        # สร้าง default settings
        self.after(200, self.save_settings)
   def _clear_all_tts_channels(self):
    """Clear all TTS channels without saving"""
    channels_to_remove = list(self.tts_channel_widgets.keys())
    for channel_id in channels_to_remove:
        self._remove_tts_channel_silent(channel_id)
   def _add_tts_channel_from_config(self, config, channel_id):
    """Add TTS channel from saved config without auto-save"""
    try:
        # Ensure we have the channel scroll frame
        if not hasattr(self, 'channel_scroll_frame'):
            self.log_message_gui(f"Channel scroll frame not ready, skipping {channel_id}", level="WARNING")
            return
        # Create channel frame
        channel_frame = ctk.CTkFrame(self.channel_scroll_frame, border_width=1)
        channel_frame.pack(pady=5, padx=5, fill="x", expand=True)
        channel_frame.grid_columnconfigure(1, weight=1)
        self.tts_channel_widgets[channel_id] = {'frame': channel_frame}
        self.tts_channel_configs[channel_id] = config.copy()
        self.tts_channel_configs[channel_id]['id'] = channel_id
        # Build channel UI
        self._build_channel_ui(channel_id, channel_frame)
        self.log_message_gui(f"Loaded channel: {channel_id}", level="DEBUG")
    except Exception as e:
        self.log_message_gui(f"Error creating channel {channel_id}: {e}", level="ERROR")
   def _remove_tts_channel_silent(self, channel_id_str):
    """Remove TTS channel without saving"""
    if channel_id_str in self.tts_channel_widgets:
        try:
            self.tts_channel_widgets[channel_id_str]['frame'].destroy()
        except:
            pass
        del self.tts_channel_widgets[channel_id_str]
    if channel_id_str in self.tts_channel_configs:
        del self.tts_channel_configs[channel_id_str]
    # Remove quota widgets
    if channel_id_str in self.quota_widgets:
        del self.quota_widgets[channel_id_str]
   def _build_channel_ui(self, channel_id_str, channel_frame):
    """Build UI for TTS channel"""
    try:
        config = self.tts_channel_configs[channel_id_str]
        # Header frame
        header_frame = ctk.CTkFrame(channel_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=4, padx=5, pady=(5,0), sticky="ew")
        ctk.CTkLabel(header_frame, text=channel_id_str, font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkButton(header_frame, text="Remove", width=60, fg_color="red", hover_color="darkred",
                     command=lambda cid=channel_id_str: self.remove_tts_channel(cid)).pack(side="right", padx=5)
        ctk.CTkButton(header_frame, text="Test Voice", width=80,
                     command=lambda cid=channel_id_str: self.test_selected_voice(cid)).pack(side="right", padx=5)
        # Engine selection
        ctk.CTkLabel(channel_frame, text="Engine:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        engines = []
        if GOOGLE_TTS_AVAILABLE: engines.append("Google Cloud TTS")
        if GTTS_AVAILABLE: engines.append("gTTS")
        engines.extend(["Gemini (Not Impl)", "Copilot (Not Impl)", "ChatGPT (Not Impl)"])
        if not engines: engines = ["None Available"]
        engine_var = tk.StringVar(value=config.get('engine', 'gTTS'))
        engine_var.trace('w', lambda *args: self.on_engine_change(channel_id_str, engine_var.get()))
        engine_menu = ctk.CTkOptionMenu(channel_frame, variable=engine_var, values=engines, width=180,
                                   command=lambda value, cid=channel_id_str: self.update_channel_options(cid, value))
        engine_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.tts_channel_widgets[channel_id_str]['engine_var'] = engine_var
        self.tts_channel_widgets[channel_id_str]['engine_menu'] = engine_menu
        # Options frame (dynamic based on engine)
        options_frame = ctk.CTkFrame(channel_frame, fg_color="transparent")
        options_frame.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")
        self.tts_channel_widgets[channel_id_str]['options_frame'] = options_frame
        # Setup options based on engine
        self.update_channel_options(channel_id_str, engine_var.get())
    except Exception as e:
        self.log_message_gui(f"Error building UI for {channel_id_str}: {e}", level="ERROR")
   def initialize_all_systems(self):
    """Initialize all systems with proper error handling"""
    try:
        # Initialize emotion system if needed
        emotion_mode = getattr(self, 'emotion_mode_var', tk.StringVar(value='normal')).get()
        if emotion_mode in ['auto_simple', 'auto_advanced']:
            self.initialize_emotion_system()
        # Initialize quota system if output folder is set
        if self.output_folder.get():
            self.initialize_quota_system()
        # Validate and repair settings if needed
        self.validate_and_repair_settings()
        self.log_message_gui("All systems initialized successfully", level="INFO")
        return True
    except Exception as e:
        self.log_message_gui(f"Error during system initialization: {e}", level="ERROR")
        return False
   def initialize_emotion_system(self):
    """Initialize emotion system with proper error handling"""
    try:
        if not hasattr(self, 'emotion_analyzer') or not self.emotion_analyzer:
            self.emotion_analyzer = EmotionAnalyzer()
        if not hasattr(self, 'ssml_generator') or not self.ssml_generator:
            self.ssml_generator = SSMLGenerator(self.emotion_analyzer)
        self.log_message_gui("Emotion system initialized successfully", level="INFO")
        return True
    except Exception as e:
        self.log_message_gui(f"Failed to initialize emotion system: {e}", level="ERROR")
        self.emotion_analyzer = None
        self.ssml_generator = None
        return False
   def validate_and_repair_settings(self):
    """Validate and repair settings if corrupted"""
    try:
        settings_file = self.get_project_settings_file()
        if not os.path.exists(settings_file):
            return True
        # Try to read the settings file
        config = configparser.ConfigParser()
        try:
            config.read(settings_file, encoding='utf-8')
        except Exception as e:
            self.log_message_gui(f"Settings file corrupted, creating backup and new file: {e}", level="WARNING")
            # Create backup
            backup_file = f"{settings_file}.backup"
            shutil.copy2(settings_file, backup_file)
            # Create new settings
            self.save_settings()
            return True
        # Validate required sections
        required_sections = ['Paths', 'Emotion', 'GenderFrequency', 'TTSChannels', 'AutoShutdown', 'APIKeys', 'UIStates']
        repaired = False
        for section in required_sections:
            if section not in config:
                config.add_section(section)
                repaired = True
        # Validate TTS channels JSON
        if 'TTSChannels' in config:
            try:
                channels_json = config.get('TTSChannels', 'channel_configs', fallback='{}')
                json.loads(channels_json)
            except json.JSONDecodeError:
                config.set('TTSChannels', 'channel_configs', '{}')
                repaired = True
        if repaired:
            with open(settings_file, 'w', encoding='utf-8') as configfile:
                config.write(configfile)
            self.log_message_gui("Settings file repaired", level="INFO")
        return True
    except Exception as e:
        self.log_message_gui(f"Error validating settings: {e}", level="ERROR")
        return False
   def save_settings(self):
    """Save current settings to the project-specific settings file with enhanced error handling"""
    config = configparser.ConfigParser()
    # ใช้ project-specific settings file
    settings_file = self.get_project_settings_file()
    try:
        # สร้างโฟลเดอร์ถ้ายังไม่มี
        output_dir = os.path.dirname(settings_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            self.log_message_gui(f"Created project directory: {output_dir}", level="INFO")
        # Save Paths
        config['Paths'] = {
            'input_folder': self.input_folder.get(),
            'output_folder': self.output_folder.get(),
            'google_voice_file': self.google_tts_voice_file.get()
        }
        # Save Emotion settings
        emotion_mode = 'normal'
        if hasattr(self, 'emotion_mode_var'):
            emotion_mode = self.emotion_mode_var.get()
        config['Emotion'] = {
            'emotion_mode': emotion_mode,
            'use_auto_emotion': str(self.use_auto_emotion_var.get()),
            'use_advanced_emotion': str(self.use_advanced_emotion_var.get())
        }
        # Save Gender Frequency settings
        config['GenderFrequency'] = {
            'male_min_hz': str(self.male_min_hz.get()),
            'male_max_hz': str(self.male_max_hz.get()),
            'female_min_hz': str(self.female_min_hz.get()),
            'female_max_hz': str(self.female_max_hz.get())
        }
        # Save TTS channel configurations with enhanced error handling
        enhanced_channel_configs = {}
        try:
            for channel_id, channel_config in self.tts_channel_configs.items():
                enhanced_config = channel_config.copy()
                # เพิ่มข้อมูล quota settings
                if hasattr(self, 'quota_manager') and self.quota_manager:
                    quota_settings = {}
                    voice_name = channel_config.get('name', '')
                    if voice_name:
                        voice_type = self.quota_manager.get_voice_type(voice_name)
                        quota_settings[voice_type] = self.quota_manager.paid_features.get(voice_type, False)
                    enhanced_config['quota_settings'] = quota_settings
                enhanced_channel_configs[channel_id] = enhanced_config
            config['TTSChannels'] = {
                'channel_configs': json.dumps(enhanced_channel_configs, ensure_ascii=False)
            }
        except Exception as e:
            self.log_message_gui(f"Error saving TTS channels: {e}", level="ERROR")
            config['TTSChannels'] = {'channel_configs': '{}'}
        # Save Auto shutdown settings
        config['AutoShutdown'] = {
            'frequency': str(self.auto_shutdown_freq.get()),
            'transcription': str(self.auto_shutdown_trans.get()),
            'translation': str(self.auto_shutdown_translate.get()),
            'synthesis': str(self.auto_shutdown_synthesis.get())
        }
        # Save API Keys paths
        config['APIKeys'] = {
            'google_stt_key': self.google_stt_key_path.get(),
            'gemini_settings': self.gemini_settings_path.get(),
            'google_tts_key': self.google_tts_key_path.get()
        }
        # Save UI States
        config['UIStates'] = {}
        if hasattr(self, 'transcription_engine_var'):
            config['UIStates']['transcription_engine'] = self.transcription_engine_var.get()
        if hasattr(self, 'translation_target_lang_var'):
            config['UIStates']['translation_language'] = self.translation_target_lang_var.get()
        if hasattr(self, 'tts_output_format_var'):
            config['UIStates']['output_format'] = self.tts_output_format_var.get()
        if hasattr(self, 'tts_input_file_var'):
            config['UIStates']['tts_input_file'] = self.tts_input_file_var.get()
        # Save Quota Management settings
        if hasattr(self, 'quota_manager') and self.quota_manager:
            config['QuotaManagement'] = {}
            try:
                for voice_type in GOOGLE_TTS_QUOTA_LIMITS.keys():
                    paid_enabled = self.quota_manager.paid_features.get(voice_type, False)
                    config['QuotaManagement'][f'{voice_type}_paid'] = str(paid_enabled)
                quota_settings = getattr(self, 'quota_settings', {})
                config['QuotaManagement']['show_warnings'] = str(quota_settings.get('show_warnings', True))
                config['QuotaManagement']['auto_refresh'] = str(quota_settings.get('auto_refresh', True))
            except Exception as e:
                self.log_message_gui(f"Error saving quota settings: {e}", level="ERROR")
        # Save Enhanced Emotion settings
        config['EnhancedEmotion'] = {}
        emotion_settings = getattr(self, 'emotion_settings', {})
        config['EnhancedEmotion']['config_file_path'] = emotion_settings.get('config_file_path', EMOTION_CONFIG_FILE)
        config['EnhancedEmotion']['enable_logging'] = str(emotion_settings.get('enable_logging', True))
        config['EnhancedEmotion']['auto_save_keywords'] = str(emotion_settings.get('auto_save_keywords', True))
        # บันทึกไฟล์
        with open(settings_file, 'w', encoding='utf-8') as configfile:
            config.write(configfile)
        # Log เฉพาะกรณีที่ไม่ใช่ auto-save
        if not hasattr(self, '_auto_save_in_progress'):
            self.log_message_gui(f"Project settings saved to: {settings_file}", level="DEBUG")
        # บันทึก quota data แยกต่างหาก
        if hasattr(self, 'quota_manager') and self.quota_manager:
            try:
                self.quota_manager.save_quota_data()
            except Exception as e:
                self.log_message_gui(f"Error saving quota data: {e}", level="ERROR")
        return True
    except Exception as e:
        self.log_message_gui(f"Critical error saving settings: {e}", level="ERROR")
        return False
   def auto_save_settings(self):
    """Auto-save settings without logging (เพื่อไม่ให้ log ยาวเกินไป)"""
    self._auto_save_in_progress = True
    result = self.save_settings()
    if hasattr(self, '_auto_save_in_progress'):
        delattr(self, '_auto_save_in_progress')
    return result
   def get_project_settings_file(self):
    """Get project-specific settings file path with proper validation"""
    output_dir = self.output_folder.get()
    if output_dir and output_dir.strip():
        # ตรวจสอบและสร้างโฟลเดอร์ถ้าจำเป็น
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                self.log_message_gui(f"Created project directory: {output_dir}", level="INFO")
            except Exception as e:
                self.log_message_gui(f"Could not create project directory: {e}", level="ERROR")
                return SETTINGS_FILE  # fallback to global
        project_settings = os.path.join(output_dir, SETTINGS_FILE)
        return project_settings
    else:
        # ถ้ายังไม่ได้เลือก output folder ให้ใช้ global settings
        return SETTINGS_FILE
   def schedule_auto_save(self):
    """Schedule auto-save with debouncing to prevent too frequent saves"""
    if self._suppress_auto_save:
        return
    # Cancel previous timer if exists
    if hasattr(self, '_auto_save_timer') and self._auto_save_timer:
        self.after_cancel(self._auto_save_timer)
    # Schedule new save
    self._auto_save_timer = self.after(1000, self._execute_auto_save)
   def _execute_auto_save(self):
    """Execute auto-save without logging"""
    try:
        self._auto_save_in_progress = True
        result = self.save_settings()
        if hasattr(self, '_auto_save_in_progress'):
            delattr(self, '_auto_save_in_progress')
        return result
    except Exception as e:
        self.log_message_gui(f"Auto-save failed: {e}", level="ERROR")
        return False
   def auto_save_settings(self):
    """Auto-save settings using scheduled approach"""
    self.schedule_auto_save()
   def add_tts_channel_silent(self, config, channel_id=None):
    """เพิ่ม TTS channel โดยไม่บันทึก settings (ใช้ตอน load)"""
    if channel_id:
        channel_id_str = channel_id
    else:
        # หาหมายเลข channel ที่ว่างอยู่
        existing_numbers = []
        for cid in self.tts_channel_widgets.keys():
            try:
                num = int(cid.split()[-1])
                existing_numbers.append(num)
            except:
                pass
        next_number = 1
        while next_number in existing_numbers:
            next_number += 1
        channel_id_str = f"Channel {next_number}"
    # เรียก add_tts_channel ปกติแต่ไม่บันทึก
    self._suppress_auto_save = True
    self.add_tts_channel(config)
    if hasattr(self, '_suppress_auto_save'):
        delattr(self, '_suppress_auto_save')
   def remove_tts_channel_silent(self, channel_id_str):
    """ลบ TTS channel โดยไม่บันทึก settings (ใช้ตอน load)"""
    if channel_id_str in self.tts_channel_widgets:
        self.tts_channel_widgets[channel_id_str]['frame'].destroy()
        del self.tts_channel_widgets[channel_id_str]
        del self.tts_channel_configs[channel_id_str]
        # ลบ quota widgets
        if channel_id_str in self.quota_widgets:
            del self.quota_widgets[channel_id_str]
   def validate_and_repair_settings(self):
    """ตรวจสอบและซ่อมแซม settings ที่เสียหาย"""
    try:
        settings_file = self.get_project_settings_file()
        if not os.path.exists(settings_file):
            return True
        # ลองอ่านไฟล์
        config = configparser.ConfigParser()
        config.read(settings_file, encoding='utf-8')
        # ตรวจสอบ sections ที่จำเป็น
        required_sections = ['Paths', 'Emotion', 'GenderFrequency', 'TTSChannels']
        repaired = False
        for section in required_sections:
            if section not in config:
                config.add_section(section)
                repaired = True
        # ซ่อมแซม TTS channels ถ้าเสียหาย
        if 'TTSChannels' in config:
            try:
                channels_json = config.get('TTSChannels', 'channel_configs', fallback='{}')
                json.loads(channels_json)  # ทดสอบว่า JSON ถูกต้องหรือไม่
            except json.JSONDecodeError:
                config.set('TTSChannels', 'channel_configs', '{}')
                repaired = True
        if repaired:
            with open(settings_file, 'w', encoding='utf-8') as configfile:
                config.write(configfile)
            self.log_message_gui("Settings file repaired", level="INFO")
        return True
    except Exception as e:
        self.log_message_gui(f"Error validating settings: {e}", level="ERROR")
        return False
# ===============================================
# UTILITY METHODS
# ===============================================
   def update_folder_path(self, var, entry_widget, title):
        """Update folder path with enhanced auto-save and project switching"""
        folder = select_folder(entry_widget, title)
        if folder:
            old_output = self.output_folder.get()
            var.set(folder)
            # ถ้าตัวแปรที่เปลี่ยนคือ Output Folder (ถือเป็น Project Folder)
            if var == self.output_folder and folder != old_output:
                self.log_message_gui(f"Project folder set to: {folder}", level="INFO")
                # ถ้ายังไม่มี Input folder ให้ตั้งเป็นอันเดียวกัน
                if not self.input_folder.get():
                    self.input_folder.set(folder)
                    self.input_folder_entry.delete(0, tk.END)
                    self.input_folder_entry.insert(0, folder)
                # ทำการโหลด Settings ของโปรเจกต์นี้ทันที (ถ้าไม่มีจะสร้างใหม่ตอน Save)
                self.load_settings()
                self.initialize_all_systems()
            # บันทึกการเปลี่ยนแปลงเสมอ
            self.schedule_auto_save()
            return folder
        return None
   def check_paths(self):
    """Check and validate paths with auto-save"""
    global input_folder_path, output_folder_path
    input_folder_path = self.input_folder.get()
    output_folder_path = self.output_folder.get()
    if not input_folder_path or not os.path.isdir(input_folder_path):
        messagebox.showerror("Error", "Input folder path is invalid.")
        self.log_message_gui("Input folder path is invalid.", level="ERROR")
        return False
    if not output_folder_path:
        messagebox.showerror("Error", "Output folder path must be set.")
        self.log_message_gui("Output folder path not set.", level="ERROR")
        return False
    try:
        if not os.path.isdir(output_folder_path):
            os.makedirs(output_folder_path, exist_ok=True)
            self.log_message_gui(f"Created output folder: {output_folder_path}", level="INFO")
            # Initialize quota system for the new output folder
            self.initialize_quota_system()
            # Auto-save settings หลังสร้าง folder
            self.auto_save_settings()
    except Exception as e:
        messagebox.showerror("Error", f"Could not create output folder:\n{output_folder_path}\n{e}")
        self.log_message_gui(f"Failed to create output folder: {e}", level="ERROR")
        return False
    return True
   def log_message_gui(self, message, level="INFO"):
    """Log message to GUI with auto-save for important events"""
    log_message(self.log_text, message, level=level)
    # Auto-save สำหรับเหตุการณ์สำคัญ
    if level in ["ERROR", "CRITICAL"] and hasattr(self, 'output_folder') and self.output_folder.get():
        try:
            self.auto_save_settings()
        except:
            pass  # ไม่ให้ error ใน auto-save ทำให้ app crash
   def update_status(self, message):
    """Update status with auto-save for completion events"""
    processing_queue.put(("status", message))
    # Auto-save เมื่องานเสร็จสิ้น
    if any(keyword in str(message).lower() for keyword in ["complete", "finished", "done"]):
        try:
            self.auto_save_settings()
        except:
            pass
   def update_progress(self, value):
    """Update progress bar"""
    processing_queue.put(("progress", value))
   def save_frequency_settings(self):
    """Save frequency range settings with validation"""
    try:
        male_min = self.male_min_hz.get()
        male_max = self.male_max_hz.get()
        female_min = self.female_min_hz.get()
        female_max = self.female_max_hz.get()
        # Validation
        if male_min >= male_max:
            messagebox.showerror("Error", "Male minimum frequency must be less than maximum")
            return False
        if female_min >= female_max:
            messagebox.showerror("Error", "Female minimum frequency must be less than maximum")
            return False
        if male_max >= female_min:
            messagebox.showwarning("Warning", "Male and female frequency ranges overlap")
        if save_gender_frequency_settings(male_min, male_max, female_min, female_max):
            # Auto-save main settings
            self.auto_save_settings()
            messagebox.showinfo("Settings Saved", "Gender frequency range settings saved successfully!")
            self.log_message_gui(f"Saved frequency settings: Male({male_min}-{male_max}Hz), Female({female_min}-{female_max}Hz)", level="INFO")
            return True
        else:
            messagebox.showerror("Error", "Failed to save frequency settings")
            return False
    except Exception as e:
        messagebox.showerror("Error", f"Invalid frequency values: {e}")
        return False
   def on_male_max_change(self, *args):
    """When male max Hz changes, adjust female min Hz to match with auto-save"""
    try:
        male_max = self.male_max_hz.get()
        self.female_min_hz.set(male_max + 1)
        # Auto-save เมื่อมีการเปลี่ยนแปลง
        self.after(500, self.auto_save_settings)  # Delay เล็กน้อยเพื่อไม่ให้ save บ่อยเกินไป
    except Exception as e:
        self.log_message_gui(f"Error updating frequency ranges: {e}", level="ERROR")
   def on_female_min_change(self, *args):
    """When female min Hz changes, adjust male max Hz to match with auto-save"""
    try:
        female_min = self.female_min_hz.get()
        if female_min > 1:
            self.male_max_hz.set(female_min - 1)
        # Auto-save เมื่อมีการเปลี่ยนแปลง
        self.after(500, self.auto_save_settings)  # Delay เล็กน้อยเพื่อไม่ให้ save บ่อยเกินไป
    except Exception as e:
        self.log_message_gui(f"Error updating frequency ranges: {e}", level="ERROR")
   def update_time_display(self):
    """Update time tracking display during processing"""
    if self.current_task_name and self.output_folder.get():
        try:
            total_time, session_time = update_task_timer(self.current_task_name, self.output_folder.get())
            # คำนวณ progress จาก progress bar
            current_progress = self.progress_bar.get()
            estimated_remaining = estimate_completion_time(current_progress, total_time) if current_progress > 0 else 0
            # สร้างข้อความแสดงผล
            elapsed_text = f"ประมวลผลไปแล้วทั้งหมดเป็นเวลา {format_time_duration(total_time)}"
            eta_text = f"คาดการณ์เวลาที่จะแล้วเสร็จ {format_time_duration(estimated_remaining)}"
            time_text = f"{elapsed_text}\n{eta_text}"
            # Update appropriate time label based on current task
            if hasattr(self, 'freq_time_label') and self.current_task_name == "frequency_analysis":
                self.freq_time_label.configure(text=time_text, text_color="green")
            elif hasattr(self, 'trans_time_label') and self.current_task_name == "transcription":
                self.trans_time_label.configure(text=time_text, text_color="green")
            elif hasattr(self, 'translate_time_label') and self.current_task_name == "translation":
                self.translate_time_label.configure(text=time_text, text_color="green")
            elif hasattr(self, 'synthesis_time_label') and self.current_task_name == "synthesis":
                self.synthesis_time_label.configure(text=time_text, text_color="green")
        except Exception as e:
            pass
    else:
        # เมื่อไม่มี task ให้แสดงข้อความเริ่มต้น
        if hasattr(self, 'freq_time_label'):
            self.freq_time_label.configure(text="Time tracking will appear here during processing", text_color="gray")
        if hasattr(self, 'trans_time_label'):
            self.trans_time_label.configure(text="Time tracking will appear here during processing", text_color="gray")
        if hasattr(self, 'translate_time_label'):
            self.translate_time_label.configure(text="Time tracking will appear here during processing", text_color="gray")
        if hasattr(self, 'synthesis_time_label'):
            self.synthesis_time_label.configure(text="Time tracking will appear here during processing", text_color="gray")
    # Schedule next update
    self.after(1000, self.update_time_display)  # Update every second
   def on_closing(self):
       """Handle the window closing event with comprehensive cleanup and auto-save"""
       try:
           self.log_message_gui("Shutting down application...", level="INFO")
        # Stop any running tasks
           if hasattr(self, 'current_task_thread') and self.current_task_thread and self.current_task_thread.is_alive():
            self.log_message_gui("Stopping current task...", level="INFO")
            stop_processing_flag.set()
            # Wait briefly for task to stop
            self.current_task_thread.join(timeout=3.0)
        # Cancel any pending auto-save timers
           if hasattr(self, '_auto_save_timer') and self._auto_save_timer:
            self.after_cancel(self._auto_save_timer)
           if hasattr(self, 'time_update_timer') and self.time_update_timer:
            self.after_cancel(self.time_update_timer)
        # Save all settings
           self.log_message_gui("Saving all settings before closing...", level="INFO")
           if not self.save_settings():
            self.log_message_gui("Warning: Some settings may not have been saved", level="WARNING")
        # Save quota data
           if hasattr(self, 'quota_manager') and self.quota_manager:
            try:
                self.quota_manager.save_quota_data()
                self.log_message_gui("Quota data saved", level="INFO")
            except Exception as e:
                self.log_message_gui(f"Error saving quota data: {e}", level="ERROR")
        # Save emotion config
           if hasattr(self, 'emotion_analyzer') and self.emotion_analyzer:
            try:
                self.emotion_analyzer.save_emotion_config()
                self.log_message_gui("Emotion config saved", level="INFO")
            except Exception as e:
                self.log_message_gui(f"Error saving emotion config: {e}", level="ERROR")
        # Show final summary
           self._show_final_summary()
           self.log_message_gui("Application shutdown completed successfully.", level="INFO")
        # Destroy the window
           self.destroy()
       except Exception as e:
           self.log_message_gui(f"Error during application shutdown: {e}", level="ERROR")
           try:
               self.destroy()
           except:
               pass
   def _show_final_summary(self):
    """Show final usage summary"""
    try:
        # Show quota summary
        if hasattr(self, 'quota_manager') and self.quota_manager:
            quota_summary = self.quota_manager.get_usage_summary()
            self.log_message_gui("=== Final Quota Summary ===", level="INFO")
            for voice_type, info in quota_summary.items():
                if info['current_usage'] > 0:
                    self.log_message_gui(f"{info['name']}: {info['current_usage']:,} {info['unit']} used this month", level="INFO")
        # Show time tracking summary
        if self.output_folder.get():
            time_data = load_time_tracking_data(self.output_folder.get())
            if time_data and 'accumulated_times' in time_data:
                self.log_message_gui("=== Total Time Spent ===", level="INFO")
                for task_name, total_time in time_data['accumulated_times'].items():
                    formatted_time = format_time_duration(total_time)
                    self.log_message_gui(f"{task_name.replace('_', ' ').title()}: {formatted_time}", level="INFO")
    except Exception as e:
        self.log_message_gui(f"Error showing final summary: {e}", level="ERROR")
   def create_enhanced_emotion_config(self):
    """Create enhanced emotion configuration file"""
    try:
        if not self.output_folder.get():
            return False
        config_path = os.path.join(self.output_folder.get(), EMOTION_ENHANCED_CONFIG_FILE)
        enhanced_config = {
            'version': '2.0',
            'last_updated': datetime.datetime.now().isoformat(),
            'thai_emotion_compatibility': THAI_EMOTION_COMPATIBILITY,
            'current_settings': {
                'emotion_mode': getattr(self, 'emotion_mode_var', tk.StringVar(value='normal')).get(),
                'use_auto_emotion': self.use_auto_emotion_var.get(),
                'use_advanced_emotion': self.use_advanced_emotion_var.get()
            }
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        self.log_message_gui(f"Error creating enhanced emotion config: {e}", level="ERROR")
        return False
   def show_startup_summary(self):
    """Show startup summary with system status"""
    try:
        summary_lines = []
        summary_lines.append(f"=== {APP_NAME} v{APP_VERSION} ===")
        summary_lines.append("")
        # System status
        emotion_status = "✓" if hasattr(self, 'emotion_analyzer') and self.emotion_analyzer else "✗"
        quota_status = "✓" if hasattr(self, 'quota_manager') and self.quota_manager else "✗"
        summary_lines.append("System Status:")
        summary_lines.append(f"  Emotion Analysis: {emotion_status}")
        summary_lines.append(f"  Quota Management: {quota_status}")
        summary_lines.append(f"  TTS Channels: {len(self.tts_channel_configs)}")
        summary_lines.append("")
        # Available engines
        engines = []
        if WHISPER_AVAILABLE:
            engines.append("Whisper")
        if GOOGLE_STT_AVAILABLE:
            engines.append("Google STT")
        if GEMINI_AVAILABLE:
            engines.append("Gemini")
        if GOOGLE_TTS_AVAILABLE:
            engines.append("Google TTS")
        if GTTS_AVAILABLE:
            engines.append("gTTS")
        summary_lines.append(f"Available Engines: {', '.join(engines) if engines else 'None'}")
        summary_lines.append("")
        # Paths
        if self.input_folder.get():
            summary_lines.append(f"Input Folder: {self.input_folder.get()}")
        if self.output_folder.get():
            summary_lines.append(f"Output Folder: {self.output_folder.get()}")
        summary_text = "\n".join(summary_lines)
        self.log_message_gui(summary_text, level="INFO")
    except Exception as e:
        self.log_message_gui(f"Error generating startup summary: {e}", level="WARNING")
# ===============================================
# PROGRESS MANAGEMENT METHODS
# ===============================================
   def clear_all_progress(self):
    """Clear all progress tracking files with auto-save"""
    if not self.output_folder.get():
        messagebox.showwarning("Warning", "Please set output folder first.")
        return
    result = messagebox.askyesno("Clear Progress", 
                          "This will clear all progress tracking data AND time tracking data.\n"
                          "You will need to restart any interrupted tasks from the beginning.\n\n"
                          "Are you sure you want to continue?")
    if result:
        try:
            progress_files = [
                FREQUENCY_PROGRESS_FILE,
                TRANSCRIPTION_PROGRESS_FILE,
                TRANSLATION_PROGRESS_FILE,
                SYNTHESIS_PROGRESS_FILE
            ]
            cleared_count = 0
            for progress_file in progress_files:
                if clear_progress(self.output_folder.get(), progress_file):
                    cleared_count += 1
            # Clear time tracking data
            time_file_path = os.path.join(self.output_folder.get(), TIME_TRACKING_FILE)
            if os.path.exists(time_file_path):
                os.remove(time_file_path)
                cleared_count += 1
                self.log_message_gui("Cleared time tracking data", level="INFO")
            # Auto-save settings หลังลบ progress
            self.auto_save_settings()
            messagebox.showinfo("Progress Cleared", 
                         f"Cleared {cleared_count} progress and time tracking files.")
            self.log_message_gui(f"Cleared {cleared_count} progress and time tracking files.", level="INFO")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear progress files:\n{e}")
            self.log_message_gui(f"Failed to clear progress files: {e}", level="ERROR")
   def show_progress_status(self):
    """Show current progress status with detailed information"""
    if not self.output_folder.get():
        messagebox.showwarning("Warning", "Please set output folder first.")
        return
    progress_info = []
    progress_files = {
        "Frequency Analysis": FREQUENCY_PROGRESS_FILE,
        "Transcription": TRANSCRIPTION_PROGRESS_FILE,
        "Translation": TRANSLATION_PROGRESS_FILE,
        "Speech Synthesis": SYNTHESIS_PROGRESS_FILE
    }
    total_progress_found = 0
    for task_name, progress_file in progress_files.items():
        progress_data = load_progress(self.output_folder.get(), progress_file)
        if progress_data:
            total_progress_found += 1
            last_updated = progress_data.get('last_updated', 'Unknown')
            if 'processed_files' in progress_data:
                # File-based progress (frequency analysis, transcription)
                processed = len(progress_data.get('processed_files', []))
                total = progress_data.get('total_files', 0)
                percentage = (processed / total * 100) if total > 0 else 0
                progress_info.append(f"{task_name}: {processed}/{total} files ({percentage:.1f}%)")
                progress_info.append(f"  Last updated: {last_updated}")
            elif 'processed_lines' in progress_data:
                # Line-based progress (translation, synthesis)
                processed = progress_data.get('processed_lines', 0)
                total = progress_data.get('total_lines', 0)
                percentage = (processed / total * 100) if total > 0 else 0
                progress_info.append(f"{task_name}: {processed}/{total} lines ({percentage:.1f}%)")
                progress_info.append(f"  Last updated: {last_updated}")
        else:
            progress_info.append(f"{task_name}: No progress data")
    # แสดง time tracking status
    time_data = load_time_tracking_data(self.output_folder.get())
    if time_data and 'accumulated_times' in time_data:
        progress_info.append("\n=== Time Tracking ===")
        last_updated = time_data.get('last_updated', 'Unknown')
        progress_info.append(f"Last updated: {last_updated}")
        total_time_all_tasks = 0
        for task_name, total_time in time_data['accumulated_times'].items():
            formatted_time = format_time_duration(total_time)
            progress_info.append(f"{task_name.replace('_', ' ').title()}: {formatted_time}")
            total_time_all_tasks += total_time
        if total_time_all_tasks > 0:
            progress_info.append(f"\nTotal time spent: {format_time_duration(total_time_all_tasks)}")
    else:
        progress_info.append("\n=== Time Tracking ===")
        progress_info.append("No time tracking data")
    # แสดงข้อมูลเพิ่มเติม
    if total_progress_found > 0:
        progress_info.insert(0, f"=== Progress Status (Found {total_progress_found} active tasks) ===")
    else:
        progress_info.insert(0, "=== Progress Status (No active tasks) ===")
    # แสดงข้อมูล current task
    if hasattr(self, 'current_task_name') and self.current_task_name:
        progress_info.append(f"\nCurrently running: {self.current_task_name.replace('_', ' ').title()}")
    status_text = "\n".join(progress_info)
    # สร้าง dialog ที่ใหญ่ขึ้นสำหรับข้อมูลเยอะ
    dialog = ctk.CTkToplevel(self)
    dialog.title("Progress & Time Status")
    dialog.geometry("600x500")
    dialog.transient(self)
    dialog.grab_set()
    # Text widget แทน messagebox
    text_widget = ctk.CTkTextbox(dialog, wrap="word")
    text_widget.pack(fill="both", expand=True, padx=10, pady=10)
    text_widget.insert("1.0", status_text)
    text_widget.configure(state="disabled")
    # Close button
    close_button = ctk.CTkButton(dialog, text="Close", command=dialog.destroy)
    close_button.pack(pady=10)
   def backup_current_progress(self):
    """สำรองข้อมูล progress ปัจจุบัน"""
    if not self.output_folder.get():
        messagebox.showwarning("Warning", "Please set output folder first.")
        return False
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.output_folder.get(), "progress_backup")
        timestamped_backup_dir = os.path.join(backup_dir, f"progress_backup_{timestamp}")
        os.makedirs(timestamped_backup_dir, exist_ok=True)
        progress_files = [
            FREQUENCY_PROGRESS_FILE,
            TRANSCRIPTION_PROGRESS_FILE,
            TRANSLATION_PROGRESS_FILE,
            SYNTHESIS_PROGRESS_FILE,
            TIME_TRACKING_FILE
        ]
        backed_up_files = []
        for filename in progress_files:
            source_path = os.path.join(self.output_folder.get(), filename)
            if os.path.exists(source_path):
                dest_path = os.path.join(timestamped_backup_dir, filename)
                shutil.copy2(source_path, dest_path)
                backed_up_files.append(filename)
        if backed_up_files:
            # สร้างไฟล์ข้อมูล backup
            backup_info = {
                'timestamp': timestamp,
                'backed_up_files': backed_up_files,
                'project_folder': self.output_folder.get(),
                'backup_type': 'progress_data'
            }
            backup_info_path = os.path.join(timestamped_backup_dir, "backup_info.json")
            with open(backup_info_path, 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, ensure_ascii=False, indent=2)
            self.log_message_gui(f"Progress backup created: {timestamped_backup_dir}", level="INFO")
            messagebox.showinfo("Backup Complete", 
                              f"Progress backup created successfully!\n\n"
                              f"Location: {timestamped_backup_dir}\n"
                              f"Files backed up: {len(backed_up_files)}")
            return True
        else:
            messagebox.showinfo("No Progress Data", "No progress data found to backup.")
            return False
    except Exception as e:
        error_msg = f"Error creating progress backup: {e}"
        self.log_message_gui(error_msg, level="ERROR")
        messagebox.showerror("Backup Error", error_msg)
        return False
   def restore_progress_from_backup(self):
    """กู้คืนข้อมูล progress จาก backup"""
    if not self.output_folder.get():
        messagebox.showwarning("Warning", "Please set output folder first.")
        return False
    backup_dir = os.path.join(self.output_folder.get(), "progress_backup")
    if not os.path.exists(backup_dir):
        messagebox.showinfo("No Backups", "No progress backups found in this project.")
        return False
    # หารายการ backup ที่มี
    backup_folders = []
    for item in os.listdir(backup_dir):
        item_path = os.path.join(backup_dir, item)
        if os.path.isdir(item_path) and item.startswith("progress_backup_"):
            backup_info_path = os.path.join(item_path, "backup_info.json")
            if os.path.exists(backup_info_path):
                try:
                    with open(backup_info_path, 'r', encoding='utf-8') as f:
                        backup_info = json.load(f)
                    backup_folders.append({
                        'folder': item,
                        'path': item_path,
                        'timestamp': backup_info.get('timestamp', 'Unknown'),
                        'files': backup_info.get('backed_up_files', [])
                    })
                except:
                    pass
    if not backup_folders:
        messagebox.showinfo("No Valid Backups", "No valid progress backups found.")
        return False
    # เรียงตาม timestamp
    backup_folders.sort(key=lambda x: x['timestamp'], reverse=True)
    # สร้าง dialog เลือก backup
    dialog = ctk.CTkToplevel(self)
    dialog.title("Restore Progress Backup")
    dialog.geometry("500x400")
    dialog.transient(self)
    dialog.grab_set()
    ctk.CTkLabel(dialog, text="Select backup to restore:", font=ctk.CTkFont(weight="bold")).pack(pady=10)
    # Listbox สำหรับแสดงรายการ backup
    listbox_frame = ctk.CTkFrame(dialog)
    listbox_frame.pack(fill="both", expand=True, padx=10, pady=10)
    backup_listbox = tk.Listbox(listbox_frame, height=10)
    backup_listbox.pack(fill="both", expand=True, padx=5, pady=5)
    selected_backup = None
    for i, backup in enumerate(backup_folders):
        display_text = f"{backup['timestamp']} ({len(backup['files'])} files)"
        backup_listbox.insert(tk.END, display_text)
    def on_restore():
        nonlocal selected_backup
        selection = backup_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a backup to restore.")
            return
        selected_backup = backup_folders[selection[0]]
        dialog.destroy()
    def on_cancel():
        nonlocal selected_backup
        selected_backup = None
        dialog.destroy()
    # Buttons
    button_frame = ctk.CTkFrame(dialog)
    button_frame.pack(fill="x", padx=10, pady=10)
    ctk.CTkButton(button_frame, text="Restore", command=on_restore).pack(side="right", padx=5)
    ctk.CTkButton(button_frame, text="Cancel", command=on_cancel).pack(side="right", padx=5)
    dialog.wait_window()
    if selected_backup:
        try:
            # คำเตือนก่อนกู้คืน
            result = messagebox.askyesno("Confirm Restore", 
                                       f"This will overwrite current progress data with backup from:\n"
                                       f"{selected_backup['timestamp']}\n\n"
                                       f"Files to restore: {len(selected_backup['files'])}\n\n"
                                       f"Continue?")
            if not result:
                return False
            # กู้คืนไฟล์
            restored_files = []
            for filename in selected_backup['files']:
                source_path = os.path.join(selected_backup['path'], filename)
                dest_path = os.path.join(self.output_folder.get(), filename)
                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
                    restored_files.append(filename)
            if restored_files:
                self.log_message_gui(f"Restored {len(restored_files)} progress files from backup", level="INFO")
                messagebox.showinfo("Restore Complete", 
                                  f"Progress data restored successfully!\n\n"
                                  f"Restored files: {len(restored_files)}\n"
                                  f"From backup: {selected_backup['timestamp']}")
                return True
            else:
                messagebox.showerror("Restore Failed", "No files were restored.")
                return False
        except Exception as e:
            error_msg = f"Error restoring progress backup: {e}"
            self.log_message_gui(error_msg, level="ERROR")
            messagebox.showerror("Restore Error", error_msg)
            return False
    return False
# ===============================================
# TASK CONTROL METHODS
# ===============================================
   def start_task(self, task_function, *args):
    """Start a task with enhanced validation and auto-save"""
    try:
        # Check if another task is running
        if self.current_task_thread and self.current_task_thread.is_alive():
            messagebox.showwarning("Busy", "Another process is already running. Please wait for it to complete or stop it first.")
            return False
        # Validate paths
        if not self.check_paths():
            return False
        # Auto-save settings before starting task
        try:
            if not self.save_settings():
                result = messagebox.askyesno("Warning", 
                                           "Could not save current settings before starting task.\n\n"
                                           "Do you want to continue anyway?")
                if not result:
                    return False
        except Exception as e:
            self.log_message_gui(f"Warning: Could not save settings before starting task: {e}", level="WARNING")
        # Reset processing state
        stop_processing_flag.clear()
        self.progress_bar.set(0)
        self.stop_button.configure(state="normal")
        self.status_label.configure(text="Starting task...")
        # Start task thread
        self.current_task_thread = threading.Thread(target=task_function, args=args, daemon=True)
        self.current_task_thread.start()
        # Log task start
        task_name = getattr(task_function, '__name__', 'Unknown Task')
        self.log_message_gui(f"Task started: {task_name}", level="INFO")
        return True
    except Exception as e:
        self.log_message_gui(f"Error starting task: {e}", level="ERROR")
        messagebox.showerror("Task Error", f"Failed to start task:\n{e}")
        return False
   def stop_current_task(self):
    """Stop current task with auto-save"""
    if self.current_task_thread and self.current_task_thread.is_alive():
        stop_processing_flag.set()
        self.status_label.configure(text="Stop signal sent. Waiting for process to terminate...")
        self.log_message_gui("Stop signal sent by user.", level="WARNING")
        # Auto-save เมื่อหยุดงาน
        try:
            self.auto_save_settings()
        except Exception as e:
            self.log_message_gui(f"Warning: Could not save settings when stopping task: {e}", level="WARNING")
        self.after(2000, self._check_thread_after_stop)
    else:
        self.stop_button.configure(state="disabled")
   def _check_thread_after_stop(self):
    """Check thread status after stop signal with cleanup"""
    if not self.current_task_thread or not self.current_task_thread.is_alive():
        self.stop_button.configure(state="disabled")
        self.update_status("Process stopped.")
        self.progress_bar.set(0)
        # Reset current task name
        self.current_task_name = None
        # Auto-save หลังจากหยุดงาน
        try:
            self.auto_save_settings()
        except Exception as e:
            self.log_message_gui(f"Warning: Could not save settings after stopping task: {e}", level="WARNING")
        self.log_message_gui("Task stopped and cleaned up", level="INFO")
   def check_queue(self):
    """Enhanced queue checking with auto-save on completion"""
    try:
        while True:
            message_type, value = processing_queue.get_nowait()
            if message_type == "status":
                status_text = str(value)
                self.status_label.configure(text=status_text)
                # Auto-save เมื่องานเสร็จสิ้นหรือเกิดข้อผิดพลาด
                if any(keyword in status_text.lower() for keyword in ["complete", "error", "cancel", "stopped", "finished"]):
                    self.stop_button.configure(state="disabled")
                    # Reset current task name
                    self.current_task_name = None
                    if "complete" in status_text.lower() or "finished" in status_text.lower():
                        self.progress_bar.set(1.0)
                        self.log_message_gui("Task completed successfully", level="INFO")
                    else:
                        self.progress_bar.set(0)
                    # Auto-save หลังจากงานเสร็จ
                    try:
                        self.auto_save_settings()
                    except Exception as e:
                        self.log_message_gui(f"Warning: Could not save settings after task completion: {e}", level="WARNING")
            elif message_type == "progress":
                try:
                    progress_value = float(value)
                    self.progress_bar.set(progress_value)
                    # Auto-save ทุก 25% ของ progress
                    if progress_value in [0.25, 0.5, 0.75, 1.0]:
                        try:
                            self.auto_save_settings()
                        except:
                            pass  # ไม่ให้ error ใน auto-save ขัดจังหวะการทำงาน
                except (ValueError, TypeError):
                    pass  # Ignore invalid progress values
            elif message_type == "log":
                self.log_message_gui(str(value[0]), level=value[1])
            processing_queue.task_done()
    except queue.Empty:
        # ตรวจสอบสถานะ thread
        if self.current_task_thread and not self.current_task_thread.is_alive():
            if not stop_processing_flag.is_set():
                self.progress_bar.set(1.0)
            self.stop_button.configure(state="disabled")
            # Reset current task name when thread finishes
            self.current_task_name = None
            self.current_task_thread = None
            # Auto-save เมื่อ thread จบ
            try:
                self.auto_save_settings()
            except:
                pass
    self.after(100, self.check_queue)
   def validate_task_prerequisites(self, task_type):
    """ตรวจสอบข้อกำหนดเบื้องต้นก่อนเริ่มงาน"""
    prerequisites_met = True
    missing_items = []
    if task_type == "frequency_analysis":
        if not self.input_folder.get() or not os.path.exists(self.input_folder.get()):
            missing_items.append("Valid input folder")
            prerequisites_met = False
        # ตรวจสอบว่ามีไฟล์เสียงในโฟลเดอร์หรือไม่
        if self.input_folder.get() and os.path.exists(self.input_folder.get()):
            audio_files = [f for f in os.listdir(self.input_folder.get()) 
                          if any(f.lower().endswith(ext) for ext in SUPPORTED_AUDIO_EXTENSIONS)]
            if not audio_files:
                missing_items.append("Audio files in input folder")
                prerequisites_met = False
    elif task_type == "transcription":
        engine = getattr(self, 'transcription_engine_var', tk.StringVar()).get()
        if engine == "Whisper":
            if not WHISPER_AVAILABLE:
                missing_items.append("Whisper library")
                prerequisites_met = False
        elif engine == "Google Cloud STT":
            if not GOOGLE_STT_AVAILABLE:
                missing_items.append("Google Cloud Speech library")
                prerequisites_met = False
            else:
                key_path = self.google_stt_key_path.get()
                if not key_path or not os.path.exists(key_path):
                    missing_items.append("Google Cloud STT key file")
                    prerequisites_met = False
        elif engine == "Gemini":
            if not GEMINI_AVAILABLE:
                missing_items.append("Gemini library")
                prerequisites_met = False
            else:
                settings_path = self.gemini_settings_path.get()
                if not settings_path or not os.path.exists(settings_path):
                    missing_items.append("Gemini settings file")
                    prerequisites_met = False
    elif task_type == "translation":
        # ตรวจสอบว่ามีไฟล์ transcription หรือไม่
        transcription_file = os.path.join(self.output_folder.get(), OUTPUT_TRANSCRIPTION_FILE)
        if not os.path.exists(transcription_file):
            missing_items.append(f"Transcription file ({OUTPUT_TRANSCRIPTION_FILE})")
            prerequisites_met = False
    elif task_type == "synthesis":
        # ตรวจสอบไฟล์ input
        input_file = getattr(self, 'tts_input_file_var', tk.StringVar()).get()
        if not input_file or not os.path.exists(input_file):
            missing_items.append("Valid input text file for synthesis")
            prerequisites_met = False
        # ตรวจสอบ TTS channels
        if not self.tts_channel_configs:
            missing_items.append("At least one TTS channel configuration")
            prerequisites_met = False
        else:
            # ตรวจสอบ Google TTS channels
            google_channels = [cid for cid, config in self.tts_channel_configs.items() 
                             if config.get('engine') == 'Google Cloud TTS']
            for channel_id in google_channels:
                config = self.tts_channel_configs[channel_id]
                key_path = config.get('google_key_path')
                if not key_path or not os.path.exists(key_path):
                    missing_items.append(f"Google TTS key file for {channel_id}")
                    prerequisites_met = False
    if not prerequisites_met:
        error_msg = f"Cannot start {task_type.replace('_', ' ')}. Missing:\n\n"
        error_msg += "\n".join([f"• {item}" for item in missing_items])
        error_msg += f"\n\nPlease configure the missing items and try again."
        messagebox.showerror("Prerequisites Not Met", error_msg)
        self.log_message_gui(f"Task validation failed for {task_type}: {', '.join(missing_items)}", level="ERROR")
    return prerequisites_met
   def pause_current_task(self):
    """หยุดชั่วคราว (สำหรับอนาคต)"""
    # Feature สำหรับอนาคต - ยังไม่ implement
    messagebox.showinfo("Feature Not Available", "Task pause/resume functionality will be available in future versions.")
   def get_task_status_summary(self):
    """ได้รับสรุปสถานะงานปัจจุบัน"""
    status_info = {
        'current_task': self.current_task_name,
        'is_running': self.current_task_thread and self.current_task_thread.is_alive(),
        'progress': self.progress_bar.get(),
        'status_text': self.status_label.cget('text'),
        'can_start_new_task': not (self.current_task_thread and self.current_task_thread.is_alive())
    }
    if self.current_task_name and self.output_folder.get():
        try:
            total_time, session_time = update_task_timer(self.current_task_name, self.output_folder.get())
            status_info['elapsed_time'] = total_time
            status_info['session_time'] = session_time
            status_info['estimated_remaining'] = estimate_completion_time(status_info['progress'], total_time)
        except:
            pass
    return status_info
   def emergency_stop_all_tasks(self):
    """หยุดงานฉุกเฉิน (สำหรับกรณีที่โปรแกรมค้าง)"""
    try:
        # ส่งสัญญาณหยุด
        stop_processing_flag.set()
        # รอ thread หยุด (สูงสุด 5 วินาที)
        if self.current_task_thread and self.current_task_thread.is_alive():
            self.current_task_thread.join(timeout=5.0)
        # Reset UI
        self.stop_button.configure(state="disabled")
        self.progress_bar.set(0)
        self.status_label.configure(text="Emergency stop completed")
        self.current_task_name = None
        self.current_task_thread = None
        # Auto-save
        self.auto_save_settings()
        self.log_message_gui("Emergency stop completed", level="WARNING")
        messagebox.showwarning("Emergency Stop", "All tasks have been forcibly stopped.")
    except Exception as e:
        self.log_message_gui(f"Error during emergency stop: {e}", level="ERROR")
# ===============================================
# TASK STARTER METHODS
# ===============================================
   def start_frequency_analysis(self):
    """Start frequency analysis with validation and auto-save"""
    # Validate prerequisites
    if not self.validate_task_prerequisites("frequency_analysis"):
        return False
    try:
        start_time = float(self.freq_start_time_entry.get())
        end_time = float(self.freq_end_time_entry.get())
        if start_time < 0 or end_time < start_time:
            raise ValueError("Invalid time range")
    except ValueError:
        messagebox.showerror("Input Error", "Invalid start/end time for frequency analysis. Please enter valid numbers (e.g., 0.0, 5.0).")
        return False
    # Get settings
    filter_non_speech = self.filter_non_speech_var.get()
    auto_shutdown = self.auto_shutdown_freq.get()
    # Get custom frequency ranges
    male_min = self.male_min_hz.get()
    male_max = self.male_max_hz.get()
    female_min = self.female_min_hz.get()
    female_max = self.female_max_hz.get()
    # Validate frequency ranges
    if male_min >= male_max or female_min >= female_max:
        messagebox.showerror("Configuration Error", "Invalid frequency ranges. Please check your gender frequency settings.")
        return False
    # Auto-save settings before starting
    if not self.auto_save_settings():
        messagebox.showwarning("Warning", "Could not save settings before starting. Continue anyway?")
    # Start time tracking
    self.current_task_name = "frequency_analysis"
    self.log_message_gui("Starting Frequency Analysis...", level="INFO")
    self.log_message_gui(f"Time range: {start_time}s - {end_time}s, Filter non-speech: {filter_non_speech}", level="INFO")
    self.log_message_gui(f"Frequency ranges - Male: {male_min}-{male_max}Hz, Female: {female_min}-{female_max}Hz", level="INFO")
    return self.start_task(analyze_frequencies_task,
                          self.input_folder.get(),
                          self.output_folder.get(),
                          start_time,
                          end_time,
                          filter_non_speech,
                          self.update_status,
                          self.update_progress,
                          self.log_text,
                          auto_shutdown,
                          male_min,
                          male_max,
                          female_min,
                          female_max)
   def start_transcription(self):
    """Start transcription with validation and auto-save"""
    # Validate prerequisites
    if not self.validate_task_prerequisites("transcription"):
        return False
    engine = self.transcription_engine_var.get()
    if engine.startswith("None") or engine.endswith("(Not Implemented)"):
        messagebox.showerror("Error", f"Transcription engine '{engine}' is not available or not implemented.")
        return False
    # Prepare options
    options = {}
    options['language'] = self.transcription_lang_var.get()
    # Engine-specific validation and setup
    if engine == "Whisper":
        model_type = getattr(self, 'whisper_model_type', tk.StringVar(value="standard")).get()
        if model_type == "standard":
            model_name = getattr(self, 'whisper_model_var', tk.StringVar(value="base")).get()
            options['whisper_model'] = model_name
            self.log_message_gui(f"Using Whisper standard model: {model_name}", level="INFO")
        elif model_type == "file":
            model_file = getattr(self, 'whisper_model_file_var', tk.StringVar()).get()
            if not model_file or not os.path.exists(model_file):
                messagebox.showerror("Error", "Whisper model file path is not set or file does not exist.")
                return False
            options['whisper_model'] = model_file
            self.log_message_gui(f"Using Whisper model file: {os.path.basename(model_file)}", level="INFO")
    elif engine == "Google Cloud STT":
        key_path = self.google_stt_key_path.get()
        if not key_path or not os.path.exists(key_path):
            messagebox.showerror("Error", "Google Cloud STT Key file path is not set or invalid.")
            return False
        options['google_key_path'] = key_path
        self.log_message_gui(f"Using Google STT key: {os.path.basename(key_path)}", level="INFO")
    elif engine == "Gemini":
        settings_path = self.gemini_settings_path.get()
        if not settings_path or not os.path.exists(settings_path):
            messagebox.showerror("Error", "Gemini Settings INI file path is not set or invalid.")
            return False
        options['gemini_settings_path'] = settings_path
        options['gemini_model_name'] = getattr(self, 'gemini_model_entry', ctk.CTkEntry(self)).get()
        options['gemini_prompt'] = self.transcription_prompt_entry.get("1.0", tk.END).strip()
        self.log_message_gui(f"Using Gemini settings: {os.path.basename(settings_path)}", level="INFO")
    # Get auto shutdown setting
    auto_shutdown = self.auto_shutdown_trans.get()
    # Auto-save current transcription settings
    if not self.auto_save_settings():
        result = messagebox.askyesno("Warning", "Could not save current settings. Continue with transcription anyway?")
        if not result:
            return False
    # Start time tracking
    self.current_task_name = "transcription"
    self.log_message_gui(f"Starting Transcription using {engine}...", level="INFO")
    self.log_message_gui(f"Language: {options['language']}, Auto shutdown: {auto_shutdown}", level="INFO")
    return self.start_task(transcribe_files_task,
                          self.input_folder.get(),
                          self.output_folder.get(),
                          engine,
                          options,
                          self.update_status,
                          self.update_progress,
                          self.log_text,
                          auto_shutdown)
   def start_translation(self):
    """Start translation with validation and auto-save"""
    # Validate prerequisites  
    if not self.validate_task_prerequisites("translation"):
        return False
    # Get and validate target language
    display_lang = self.translation_target_lang_var.get()
    target_lang_code = self.lang_display_to_code.get(display_lang)
    if not target_lang_code:
        messagebox.showerror("Error", f"Could not find language code for '{display_lang}'.")
        return False
    # Check input file exists
    input_file = os.path.join(self.output_folder.get(), OUTPUT_TRANSCRIPTION_FILE)
    if not os.path.exists(input_file):
        messagebox.showerror("Error", f"Input file not found: {OUTPUT_TRANSCRIPTION_FILE}\n\nPlease complete transcription first.")
        return False
    # Check if input file has content
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                messagebox.showerror("Error", f"Input file {OUTPUT_TRANSCRIPTION_FILE} is empty.\n\nPlease ensure transcription was completed successfully.")
                return False
        self.log_message_gui(f"Found {len(lines)} lines to translate", level="INFO")
    except Exception as e:
        messagebox.showerror("Error", f"Could not read input file: {e}")
        return False
    auto_shutdown = self.auto_shutdown_translate.get()
    # Auto-save current translation settings
    if not self.auto_save_settings():
        result = messagebox.askyesno("Warning", "Could not save current settings. Continue with translation anyway?")
        if not result:
            return False
    # Start time tracking
    self.current_task_name = "translation"
    self.log_message_gui(f"Starting Translation to {display_lang} ({target_lang_code})...", level="INFO")
    self.log_message_gui(f"Input: {len(lines)} lines, Auto shutdown: {auto_shutdown}", level="INFO")
    return self.start_task(translate_texts_task,
                          self.output_folder.get(),
                          target_lang_code,
                          self.update_status,
                          self.update_progress,
                          self.log_text,
                          auto_shutdown)
   def start_speech_synthesis(self):
    """Start speech synthesis with comprehensive validation and auto-save"""
    # Validate prerequisites
    if not self.validate_task_prerequisites("synthesis"):
        return False
    input_file = self.tts_input_file_var.get()
    if not os.path.exists(input_file):
        messagebox.showerror("Error", f"Input text file not found:\n{input_file}")
        return False
    if not self.tts_channel_configs:
        messagebox.showerror("Error", "No synthesis channels configured. Please add at least one channel.")
        return False
    # Validate emotion system configuration
    if not self.validate_emotion_selection():
        return False
    # Show emotion compatibility warnings
    self.show_emotion_compatibility_warning()
    # Initialize and validate quota system
    self.initialize_quota_system()
    quota_ok, quota_message = self.validate_google_tts_quota_before_synthesis()
    if not quota_ok:
        messagebox.showerror("Quota Error", quota_message)
        return False
    elif "warning" in quota_message.lower():
        result = messagebox.askyesno("Quota Warning", f"{quota_message}\n\nDo you want to continue?")
        if not result:
            return False
    # Show quota summary
    self.show_pre_synthesis_quota_summary()
    # Validate input file content
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                messagebox.showerror("Error", f"Input file {os.path.basename(input_file)} is empty.")
                return False
        # Count valid lines with channel markers
        valid_lines = 0
        channel_usage = {}
        for line in lines:
            _, _, _, channel_id, text = parse_output_line(line)
            if channel_id and text and channel_id.strip():
                valid_lines += 1
                channel_usage[channel_id] = channel_usage.get(channel_id, 0) + 1
        if valid_lines == 0:
            messagebox.showerror("Error", "No valid synthesis lines found in input file.\n\nExpected format: Filename|Gender|Age|Channel N|Text")
            return False
        self.log_message_gui(f"Found {valid_lines} valid lines for synthesis", level="INFO")
        self.log_message_gui(f"Channel usage: {channel_usage}", level="INFO")
        # Validate that configured channels match usage
        missing_channels = []
        for channel_id in channel_usage.keys():
            if channel_id not in self.tts_channel_configs:
                missing_channels.append(channel_id)
        if missing_channels:
            messagebox.showerror("Configuration Error", 
                               f"Missing channel configurations for: {', '.join(missing_channels)}\n\n"
                               f"Please add configurations for these channels.")
            return False
    except Exception as e:
        messagebox.showerror("Error", f"Could not analyze input file: {e}")
        return False
    # Get synthesis settings
    output_format = self.tts_output_format_var.get()
    active_configs = self.tts_channel_configs.copy()
    auto_shutdown = self.auto_shutdown_synthesis.get()
    # Get regex settings
    selected_mode_name = self.regex_mode_var.get()
    selected_mode = RegexMode[selected_mode_name]
    custom_pattern = self.custom_regex_entry.get()
    if selected_mode == RegexMode.CUSTOM and not custom_pattern:
        messagebox.showerror("Input Error", "Please specify Custom Regex pattern or change mode")
        return False
    # Get emotion settings
    mode = self.emotion_mode_var.get()
    use_auto_emotion = mode in ["auto_simple", "auto_advanced"]
    use_advanced_emotion = mode == "auto_advanced"
    # Validate emotion system if enabled
    if use_auto_emotion:
        if not hasattr(self, 'emotion_analyzer') or not self.emotion_analyzer:
            try:
                self.emotion_analyzer = EmotionAnalyzer()
                self.ssml_generator = SSMLGenerator(self.emotion_analyzer)
            except Exception as e:
                messagebox.showerror("Emotion System Error", f"Could not initialize emotion system: {e}\n\nSwitching to normal mode.")
                self.emotion_mode_var.set("normal")
                use_auto_emotion = False
                use_advanced_emotion = False
    # Auto-save all current settings before synthesis
    if not self.auto_save_settings():
        result = messagebox.askyesno("Warning", "Could not save current settings. Continue with synthesis anyway?")
        if not result:
            return False
    # Start time tracking
    self.current_task_name = "synthesis"
    # Log synthesis configuration
    self.log_message_gui(f"Starting Speech Synthesis from {os.path.basename(input_file)}...", level="INFO")
    self.log_message_gui(f"Output format: {output_format}, Channels: {len(active_configs)}", level="INFO")
    self.log_message_gui(f"Regex mode: {selected_mode.name}, Auto shutdown: {auto_shutdown}", level="INFO")
    if use_auto_emotion:
        mode_text = "Advanced (sentence splitting)" if use_advanced_emotion else "Simple (first keyword)"
        self.log_message_gui(f"Emotion analysis: {mode_text}", level="INFO")
    else:
        self.log_message_gui("Emotion analysis: Disabled (manual settings)", level="INFO")
    return self.start_task(enhanced_synthesize_speech_task,
                          input_file,
                          self.output_folder.get(),
                          None,
                          active_configs,
                          output_format,
                          self.update_status,
                          self.update_progress,
                          self.log_text,
                          auto_shutdown,
                          selected_mode,
                          custom_pattern,
                          use_auto_emotion,
                          use_advanced_emotion)
   def prepare_synthesis_summary(self):
    """เตรียมสรุปการตั้งค่าก่อนเริ่ม synthesis"""
    summary_lines = []
    summary_lines.append("=== Speech Synthesis Configuration ===")
    # Input file info
    input_file = self.tts_input_file_var.get()
    if os.path.exists(input_file):
        file_size = os.path.getsize(input_file)
        summary_lines.append(f"Input: {os.path.basename(input_file)} ({file_size:,} bytes)")
    # Channel configurations
    summary_lines.append(f"Channels: {len(self.tts_channel_configs)}")
    for channel_id, config in self.tts_channel_configs.items():
        engine = config.get('engine', 'Unknown')
        voice_name = config.get('name', 'Not set')
        summary_lines.append(f"  {channel_id}: {engine} - {voice_name}")
    # Emotion settings
    mode = self.emotion_mode_var.get()
    if mode == "normal":
        summary_lines.append("Emotion: Manual settings per channel")
    elif mode == "auto_simple":
        summary_lines.append("Emotion: Auto analysis (simple - first keyword)")
    elif mode == "auto_advanced":
        summary_lines.append("Emotion: Auto analysis (advanced - sentence splitting)")
    # Quota status for Google TTS
    if hasattr(self, 'quota_manager') and self.quota_manager:
        google_channels = [cid for cid, config in self.tts_channel_configs.items() 
                          if config.get('engine') == 'Google Cloud TTS']
        if google_channels:
            summary_lines.append("Quota status:")
            quota_summary = self.quota_manager.get_usage_summary()
            for voice_type, info in quota_summary.items():
                if info['current_usage'] > 0 or any(self.quota_manager.get_voice_type(
                    self.tts_channel_configs[cid].get('name', '')) == voice_type for cid in google_channels):
                    summary_lines.append(f"  {info['name']}: {info['current_usage']:,}/{info['limit']:,} {info['unit']} ({info['percentage']:.1f}%)")
    return "\n".join(summary_lines)
   def validate_synthesis_channels(self):
    """ตรวจสอบความถูกต้องของ TTS channels"""
    issues = []
    for channel_id, config in self.tts_channel_configs.items():
        engine = config.get('engine')
        if engine == "Google Cloud TTS":
            # ตรวจสอบ key file
            key_path = config.get('google_key_path')
            if not key_path or not os.path.exists(key_path):
                issues.append(f"{channel_id}: Missing or invalid Google TTS key file")
            # ตรวจสอบ voice
            voice_name = config.get('name')
            if not voice_name:
                issues.append(f"{channel_id}: No voice selected")
        elif engine == "gTTS":
            # ตรวจสอบ language code
            lang_code = config.get('languageCode')
            if not lang_code:
                issues.append(f"{channel_id}: No language code set")
        else:
            issues.append(f"{channel_id}: Unsupported or unimplemented engine '{engine}'")
    return issues
# ===============================================
# TRANSCRIPTION OPTIONS MANAGEMENT
# ===============================================
   def update_transcription_options(self, selected_engine):
    """Update transcription options with enhanced error handling and auto-save"""
    try:
        # Clear existing options
        if hasattr(self, 'transcription_options_frame'):
            for widget in self.transcription_options_frame.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
        # Hide prompt frame by default
        if hasattr(self, 'prompt_frame'):
            self.prompt_frame.grid_remove()
        # Auto-save engine selection
        self.schedule_auto_save()
        # Setup options based on selected engine
        if selected_engine == "Whisper" and WHISPER_AVAILABLE:
            self._setup_whisper_options()
        elif selected_engine == "Google Cloud STT" and GOOGLE_STT_AVAILABLE:
            self._setup_google_stt_options()
        elif selected_engine == "Gemini" and GEMINI_AVAILABLE:
            self._setup_gemini_options()
        elif selected_engine.startswith("Copilot"):
            self._setup_copilot_options()
        elif selected_engine.startswith("ChatGPT"):
            self._setup_chatgpt_options()
        else:
            self._setup_unavailable_engine_notice(selected_engine)
    except Exception as e:
        self.log_message_gui(f"Error updating transcription options: {e}", level="ERROR")
   def _setup_whisper_options(self):
    """Setup Whisper transcription options with auto-save"""
    whisper_frame = ctk.CTkFrame(self.transcription_options_frame, fg_color="transparent")
    whisper_frame.grid(row=0, column=0, columnspan=3, padx=0, pady=5, sticky="ew")
    whisper_frame.grid_columnconfigure(1, weight=1)
    ctk.CTkLabel(whisper_frame, text="Whisper Model:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    # Model type selection with auto-save
    self.whisper_model_type = tk.StringVar(value="standard")
    self.whisper_model_type.trace('w', lambda *args: self.on_whisper_model_type_change())
    type_frame = ctk.CTkFrame(whisper_frame, fg_color="transparent")
    type_frame.grid(row=0, column=1, padx=10, pady=5, sticky="w")
    ctk.CTkRadioButton(type_frame, text="Standard Models", 
                      variable=self.whisper_model_type, value="standard", 
                      command=self.update_whisper_model_options).pack(side="left", padx=5)
    ctk.CTkRadioButton(type_frame, text="Local Model File", 
                      variable=self.whisper_model_type, value="file", 
                      command=self.update_whisper_model_options).pack(side="left", padx=5)
    # Model options frame
    self.whisper_model_options_frame = ctk.CTkFrame(whisper_frame, fg_color="transparent")
    self.whisper_model_options_frame.grid(row=1, column=0, columnspan=3, padx=0, pady=5, sticky="ew")
    self.whisper_model_options_frame.grid_columnconfigure(1, weight=1)
    # Load saved model type
    try:
        config = configparser.ConfigParser()
        settings_file = self.get_project_settings_file()
        if os.path.exists(settings_file):
            config.read(settings_file, encoding='utf-8')
            if 'Transcription' in config:
                saved_model_type = config.get('Transcription', 'whisper_model_type', fallback='standard')
                self.whisper_model_type.set(saved_model_type)
    except:
        pass
    self.update_whisper_model_options()
   def _setup_google_stt_options(self):
    """Setup Google Cloud STT options with auto-save"""
    ctk.CTkLabel(self.transcription_options_frame, text="Google Cloud Key File (.json):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    key_frame = ctk.CTkFrame(self.transcription_options_frame, fg_color="transparent")
    key_frame.grid(row=0, column=1, columnspan=2, padx=0, pady=5, sticky="ew")
    key_frame.grid_columnconfigure(0, weight=1)
    self.google_stt_key_entry = ctk.CTkEntry(key_frame, textvariable=self.google_stt_key_path, width=300)
    self.google_stt_key_entry.grid(row=0, column=0, padx=(0,10), pady=0, sticky="ew")
    self.google_stt_key_entry.bind('<KeyRelease>', lambda e: self.after(500, self.auto_save_settings))
    ctk.CTkButton(key_frame, text="Browse...", width=80, 
                 command=self.select_google_stt_key).grid(row=0, column=1, padx=0, pady=0)
    self.google_stt_key_status = ctk.CTkLabel(key_frame, text="Key not checked", text_color="gray")
    self.google_stt_key_status.grid(row=0, column=2, padx=10, pady=0)
    ctk.CTkButton(key_frame, text="Validate Key", width=100, 
                 command=lambda: validate_google_json_key(self.google_stt_key_path.get(), self.google_stt_key_status)).grid(row=0, column=3, padx=(5,0), pady=0)
   def _setup_gemini_options(self):
    """Setup Gemini options with auto-save"""
    ctk.CTkLabel(self.transcription_options_frame, text="Gemini Settings File (.ini):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    settings_frame = ctk.CTkFrame(self.transcription_options_frame, fg_color="transparent")
    settings_frame.grid(row=0, column=1, columnspan=2, padx=0, pady=5, sticky="ew")
    settings_frame.grid_columnconfigure(0, weight=1)
    self.gemini_settings_entry = ctk.CTkEntry(settings_frame, textvariable=self.gemini_settings_path, width=300)
    self.gemini_settings_entry.grid(row=0, column=0, padx=(0,10), pady=0, sticky="ew")
    self.gemini_settings_entry.bind('<KeyRelease>', lambda e: self.after(500, self.auto_save_settings))
    ctk.CTkButton(settings_frame, text="Browse...", width=80, 
                 command=self.select_gemini_settings).grid(row=0, column=1, padx=0, pady=0)
    self.gemini_key_status = ctk.CTkLabel(settings_frame, text="Status based on INI", text_color="gray")
    self.gemini_key_status.grid(row=0, column=2, padx=10, pady=0)
    ctk.CTkButton(settings_frame, text="Check First Key", width=110, 
                 command=self.check_first_gemini_key_from_ini).grid(row=0, column=3, padx=(5,0), pady=0)
    # Model name override
    ctk.CTkLabel(self.transcription_options_frame, text="Model Name (optional, overrides INI):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    self.gemini_model_entry = ctk.CTkEntry(self.transcription_options_frame, placeholder_text="e.g., gemini-1.5-flash-latest", width=250)
    self.gemini_model_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
    self.gemini_model_entry.bind('<KeyRelease>', lambda e: self.after(500, self.auto_save_settings))
    # Show prompt frame for Gemini
    self.prompt_frame.grid()
   def _setup_copilot_options(self):
    """Setup Copilot options (placeholder)"""
    ctk.CTkLabel(self.transcription_options_frame, text="Copilot Model Name (if applicable):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    self.copilot_model_entry = ctk.CTkEntry(self.transcription_options_frame, placeholder_text="Enter model identifier", width=250)
    self.copilot_model_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")
    self.copilot_model_entry.bind('<KeyRelease>', lambda e: self.after(500, self.auto_save_settings))
    ctk.CTkLabel(self.transcription_options_frame, text="Copilot Key File/Input:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    ctk.CTkLabel(self.transcription_options_frame, text="Warning: Copilot API for direct audio transcription may require specific endpoints or setup.", 
                text_color="orange").grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="w")
   def _setup_chatgpt_options(self):
    """Setup ChatGPT options (placeholder)"""
    ctk.CTkLabel(self.transcription_options_frame, text="ChatGPT Model Name (e.g., whisper-1):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    self.chatgpt_model_entry = ctk.CTkEntry(self.transcription_options_frame, placeholder_text="Enter model identifier", width=250)
    self.chatgpt_model_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")
    self.chatgpt_model_entry.bind('<KeyRelease>', lambda e: self.after(500, self.auto_save_settings))
    ctk.CTkLabel(self.transcription_options_frame, text="OpenAI API Key:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    ctk.CTkLabel(self.transcription_options_frame, text="Note: ChatGPT transcription often uses the Whisper model via OpenAI API.", 
                text_color="yellow").grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="w")
   def _setup_unavailable_engine_notice(self, engine_name):
    """Show notice for unavailable engines"""
    ctk.CTkLabel(self.transcription_options_frame, 
                text=f"Engine '{engine_name}' is not available or not implemented.", 
                text_color="red").grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="w")
   def update_whisper_model_options(self):
    """Update Whisper model options with enhanced error handling and auto-save"""
    try:
        # Clear existing widgets
        if hasattr(self, 'whisper_model_options_frame'):
            for widget in self.whisper_model_options_frame.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
        # Get model type
        model_type = getattr(self, 'whisper_model_type', tk.StringVar(value="standard")).get()
        if model_type == "standard":
            self._setup_whisper_standard_models()
        elif model_type == "file":
            self._setup_whisper_file_model()
        else:
            # Fallback to standard
            self.whisper_model_type.set("standard")
            self._setup_whisper_standard_models()
        # Auto-save when model type changes
        self.schedule_auto_save()
    except Exception as e:
        self.log_message_gui(f"Error updating Whisper model options: {e}", level="ERROR")
   def on_whisper_model_type_change(self):
    """Handle Whisper model type change with auto-save"""
    try:
        self.update_whisper_model_options()
        self.schedule_auto_save()
    except Exception as e:
        self.log_message_gui(f"Error handling Whisper model type change: {e}", level="ERROR")
   def on_whisper_standard_model_change(self):
    """Handle Whisper standard model change with auto-save"""
    try:
        self.schedule_auto_save()
    except Exception as e:
        self.log_message_gui(f"Error handling Whisper standard model change: {e}", level="ERROR")
   def _setup_whisper_standard_models(self):
    """Setup standard Whisper models with auto-save"""
    ctk.CTkLabel(self.whisper_model_options_frame, text="Select Model:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    # Load saved model
    saved_model = "base"
    try:
        config = configparser.ConfigParser()
        settings_file = self.get_project_settings_file()
        if os.path.exists(settings_file):
            config.read(settings_file, encoding='utf-8')
            if 'Transcription' in config:
                saved_model = config.get('Transcription', 'whisper_standard_model', fallback='base')
    except:
        pass
    self.whisper_model_var = tk.StringVar(value=saved_model)
    self.whisper_model_var.trace('w', lambda *args: self.on_whisper_standard_model_change())
    models = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "turbo"]
    model_menu = ctk.CTkOptionMenu(self.whisper_model_options_frame, variable=self.whisper_model_var, values=models)
    model_menu.grid(row=0, column=1, padx=10, pady=5, sticky="w")
   def _setup_whisper_file_model(self):
    """Setup Whisper file model with auto-save"""
    ctk.CTkLabel(self.whisper_model_options_frame, text="Model File (.pt):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    model_file_frame = ctk.CTkFrame(self.whisper_model_options_frame, fg_color="transparent")
    model_file_frame.grid(row=0, column=1, columnspan=2, padx=0, pady=5, sticky="ew")
    model_file_frame.grid_columnconfigure(0, weight=1)
    self.whisper_model_file_var = tk.StringVar()
    # Load last used model file
    try:
        config = configparser.ConfigParser()
        settings_file = self.get_project_settings_file()
        if os.path.exists(settings_file):
            config.read(settings_file, encoding='utf-8')
            if 'Transcription' in config:
                last_model_file = config.get('Transcription', 'whisper_model_file', fallback='')
                if last_model_file and os.path.exists(last_model_file):
                    self.whisper_model_file_var.set(last_model_file)
    except:
        pass
    self.whisper_model_file_entry = ctk.CTkEntry(model_file_frame, textvariable=self.whisper_model_file_var, width=300)
    self.whisper_model_file_entry.grid(row=0, column=0, padx=(0,10), pady=0, sticky="ew")
    self.whisper_model_file_entry.bind('<KeyRelease>', lambda e: self.after(500, self.auto_save_settings))
    ctk.CTkButton(model_file_frame, text="Browse...", width=80, 
                 command=self.select_whisper_model_file).grid(row=0, column=1, padx=0, pady=0)
    ctk.CTkLabel(self.whisper_model_options_frame, text="Note: Select a local Whisper model file (.pt)", 
                text_color="gray").grid(row=1, column=0, columnspan=3, padx=10, pady=2, sticky="w")
   def select_whisper_model_file(self):
    """Select Whisper model file with auto-save"""
    file_selected = filedialog.askopenfilename(
        title="Select Whisper Model File",
        filetypes=(("PyTorch Model files", "*.pt"), ("All files", "*.*"))
    )
    if file_selected:
        self.whisper_model_file_var.set(file_selected)
        self.whisper_model_file_entry.delete(0, tk.END)
        self.whisper_model_file_entry.insert(0, file_selected)
        # Auto-save immediately when file selected
        self.auto_save_settings()
        self.log_message_gui(f"Selected Whisper model file: {os.path.basename(file_selected)}", level="INFO")
   def select_google_stt_key(self):
    """Select Google STT key file with auto-save"""
    file_selected = filedialog.askopenfilename(
        title="Select Google Cloud STT Key File",
        filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
    )
    if file_selected:
        self.google_stt_key_path.set(file_selected)
        self.google_stt_key_entry.delete(0, tk.END)
        self.google_stt_key_entry.insert(0, file_selected)
        # Auto-save and validate
        self.auto_save_settings()
        validate_google_json_key(file_selected, self.google_stt_key_status)
        self.log_message_gui(f"Selected Google STT key: {os.path.basename(file_selected)}", level="INFO")
   def select_gemini_settings(self):
    """Select Gemini settings file with auto-save"""
    file_selected = filedialog.askopenfilename(
        title="Select Gemini Settings File",
        filetypes=(("INI files", "*.ini"), ("All files", "*.*"))
    )
    if file_selected:
        self.gemini_settings_path.set(file_selected)
        self.gemini_settings_entry.delete(0, tk.END)
        self.gemini_settings_entry.insert(0, file_selected)
        # Auto-save and check key
        self.auto_save_settings()
        self.check_first_gemini_key_from_ini()
        self.log_message_gui(f"Selected Gemini settings: {os.path.basename(file_selected)}", level="INFO")
   def check_first_gemini_key_from_ini(self):
    """Check first Gemini key from INI file"""
    settings_path = self.gemini_settings_path.get()
    if not settings_path or not os.path.exists(settings_path):
        self.gemini_key_status.configure(text="INI File Not Found", text_color="red")
        return
    config = configparser.ConfigParser()
    try:
        config.read(settings_path, encoding='utf-8')
        api_keys = {k: v for k, v in config.items('API_KEYS')}
        if not api_keys:
            self.gemini_key_status.configure(text="No Keys in INI", text_color="red")
            return
        first_key = None
        for i in range(1, len(api_keys) + 1):
            key_name = f"api_key{i}"
            if key_name in api_keys:
                first_key = api_keys[key_name]
                break
        if not first_key and api_keys:
            first_key = next(iter(api_keys.values()))
        if first_key:
            validate_gemini_api_key(first_key, self.gemini_key_status)
        else:
            self.gemini_key_status.configure(text="No Keys Found", text_color="red")
    except Exception as e:
        self.gemini_key_status.configure(text=f"Error Reading INI: {e}", text_color="red")
   def on_whisper_model_type_change(self):
    """Handle Whisper model type change with auto-save"""
    self.update_whisper_model_options()
    # Save the model type selection
    self.after(100, self.auto_save_settings)
   def on_whisper_standard_model_change(self):
    """Handle Whisper standard model change with auto-save"""
    self.after(500, self.auto_save_settings)
   def save_transcription_settings_to_config(self):
    """Save transcription settings to config file"""
    try:
        config = configparser.ConfigParser()
        settings_file = self.get_project_settings_file()
        # Read existing config
        if os.path.exists(settings_file):
            config.read(settings_file, encoding='utf-8')
        # Add Transcription section
        if 'Transcription' not in config:
            config.add_section('Transcription')
        # Save engine
        if hasattr(self, 'transcription_engine_var'):
            config.set('Transcription', 'engine', self.transcription_engine_var.get())
        # Save language
        if hasattr(self, 'transcription_lang_var'):
            config.set('Transcription', 'language', self.transcription_lang_var.get())
        # Save Whisper settings
        if hasattr(self, 'whisper_model_type'):
            config.set('Transcription', 'whisper_model_type', self.whisper_model_type.get())
        if hasattr(self, 'whisper_model_var'):
            config.set('Transcription', 'whisper_standard_model', self.whisper_model_var.get())
        if hasattr(self, 'whisper_model_file_var'):
            config.set('Transcription', 'whisper_model_file', self.whisper_model_file_var.get())
        # Save Gemini model override
        if hasattr(self, 'gemini_model_entry'):
            config.set('Transcription', 'gemini_model_override', self.gemini_model_entry.get())
        # Save prompt
        if hasattr(self, 'transcription_prompt_entry'):
            try:
                prompt_text = self.transcription_prompt_entry.get("1.0", tk.END).strip()
                config.set('Transcription', 'prompt_text', prompt_text)
            except:
                pass
        # Write config
        with open(settings_file, 'w', encoding='utf-8') as configfile:
            config.write(configfile)
        return True
    except Exception as e:
        self.log_message_gui(f"Error saving transcription settings: {e}", level="ERROR")
        return False
   def load_transcription_settings_from_config(self):
    """Load transcription settings from config file"""
    try:
        config = configparser.ConfigParser()
        settings_file = self.get_project_settings_file()
        if not os.path.exists(settings_file):
            return
        config.read(settings_file, encoding='utf-8')
        if 'Transcription' not in config:
            return
        # Load engine
        engine = config.get('Transcription', 'engine', fallback='Whisper')
        if hasattr(self, 'transcription_engine_var'):
            self.transcription_engine_var.set(engine)
        # Load language  
        language = config.get('Transcription', 'language', fallback='en')
        if hasattr(self, 'transcription_lang_var'):
            self.transcription_lang_var.set(language)
        # Load prompt
        prompt_text = config.get('Transcription', 'prompt_text', fallback='Accurately transcribe the entire audio file, line by line, into colloquial Thai.')
        if hasattr(self, 'transcription_prompt_entry'):
            try:
                self.transcription_prompt_entry.delete("1.0", tk.END)
                self.transcription_prompt_entry.insert("1.0", prompt_text)
            except:
                pass
        self.log_message_gui("Transcription settings loaded from config", level="DEBUG")
    except Exception as e:
        self.log_message_gui(f"Error loading transcription settings: {e}", level="ERROR")
# ===============================================
# TTS CHANNEL MANAGEMENT
# ===============================================
   def select_tts_input_file(self):
    """Select TTS input file with auto-save"""
    file_selected = filedialog.askopenfilename(
        title="Select Input Text File", 
        filetypes=(("Text files", "*.txt"), ("All files", "*.*")), 
        initialdir=self.output_folder.get()
    )
    if file_selected:
        self.tts_input_file_var.set(file_selected)
        # Auto-save when input file is selected
        self.auto_save_settings()
        self.log_message_gui(f"Selected TTS input file: {os.path.basename(file_selected)}", level="INFO")
   def add_tts_channel(self, config=None):
    """Add TTS channel with enhanced error handling and auto-save"""
    try:
        # ตรวจสอบว่ามี channel_scroll_frame แล้ว
        if not hasattr(self, 'channel_scroll_frame'):
            self.log_message_gui("Channel scroll frame not ready yet", level="WARNING")
            return False
        # หาหมายเลข channel ที่ว่างอยู่
        existing_numbers = []
        for channel_id in self.tts_channel_widgets.keys():
            try:
                num = int(channel_id.split()[-1])
                existing_numbers.append(num)
            except:
                pass
        # หาหมายเลขที่เล็กที่สุดที่ยังไม่ได้ใช้
        next_number = 1
        while next_number in existing_numbers:
            next_number += 1
        channel_id_str = f"Channel {next_number}"
        # สร้าง default config ถ้าไม่มี
        if config is None:
            config = {
                'id': channel_id_str, 
                'engine': 'gTTS',
                'languageCode': 'th',
                'speed': 1.0,
                'pitch': 0.0,
                'request_delay': 20.0
            }
        else:
            config = config.copy()
            config['id'] = channel_id_str
        # เพิ่ม config ก่อน
        self.tts_channel_configs[channel_id_str] = config
        # สร้าง channel frame
        channel_frame = ctk.CTkFrame(self.channel_scroll_frame, border_width=1)
        channel_frame.pack(pady=5, padx=5, fill="x", expand=True)
        channel_frame.grid_columnconfigure(1, weight=1)
        self.tts_channel_widgets[channel_id_str] = {'frame': channel_frame}
        # Build UI
        self._build_channel_ui(channel_id_str, channel_frame)
        # Auto-save after adding channel (only if not during loading)
        if not self._suppress_auto_save:
            self.schedule_auto_save()
            self.log_message_gui(f"Added TTS Channel: {channel_id_str} ({config['engine']})", level="INFO")
        return True
    except Exception as e:
        self.log_message_gui(f"Error adding TTS channel: {e}", level="ERROR")
        return False
   def add_tts_channel_silent(self, config, channel_id=None):
    """Add TTS channel silently during loading"""
    if channel_id:
        channel_id_str = channel_id
    else:
        # หาหมายเลข channel ที่ว่างอยู่
        existing_numbers = []
        for cid in self.tts_channel_widgets.keys():
            try:
                num = int(cid.split()[-1])
                existing_numbers.append(num)
            except:
                pass
        next_number = 1
        while next_number in existing_numbers:
            next_number += 1
        channel_id_str = f"Channel {next_number}"
    # Suppress auto-save during loading
    old_suppress = self._suppress_auto_save
    self._suppress_auto_save = True
    try:
        result = self.add_tts_channel(config)
        return result
    finally:
        self._suppress_auto_save = old_suppress
   def remove_tts_channel(self, channel_id_str):
    """Remove TTS channel with auto-save"""
    if channel_id_str in self.tts_channel_widgets:
        # Confirm removal
        result = messagebox.askyesno("Remove Channel", f"Are you sure you want to remove {channel_id_str}?")
        if not result:
            return
        # Remove widgets
        self.tts_channel_widgets[channel_id_str]['frame'].destroy()
        del self.tts_channel_widgets[channel_id_str]
        del self.tts_channel_configs[channel_id_str]
        # ลบ quota widgets
        if channel_id_str in self.quota_widgets:
            del self.quota_widgets[channel_id_str]
        # Auto-save after removing channel
        self.auto_save_settings()
        self.log_message_gui(f"Removed TTS Channel: {channel_id_str}", level="INFO")
   def update_channel_options(self, channel_id_str, selected_engine):
    """Update channel options with enhanced error handling"""
    try:
        # Validate inputs
        if channel_id_str not in self.tts_channel_configs:
            self.log_message_gui(f"Channel {channel_id_str} not found in configs", level="ERROR")
            return
        if channel_id_str not in self.tts_channel_widgets:
            self.log_message_gui(f"Channel {channel_id_str} not found in widgets", level="ERROR")
            return
        # Update config
        self.tts_channel_configs[channel_id_str]['engine'] = selected_engine
        # Get options frame
        options_frame = self.tts_channel_widgets[channel_id_str].get('options_frame')
        if not options_frame:
            self.log_message_gui(f"Options frame not found for {channel_id_str}", level="ERROR")
            return
        # Clear existing options
        for widget in options_frame.winfo_children():
            try:
                widget.destroy()
            except:
                pass
        # Reset grid configuration
        options_frame.grid_columnconfigure(1, weight=0)
        options_frame.grid_columnconfigure(3, weight=0)
        config = self.tts_channel_configs[channel_id_str]
        # Setup options based on engine
        if selected_engine == "Google Cloud TTS" and GOOGLE_TTS_AVAILABLE:
            self._setup_google_tts_options(options_frame, channel_id_str, config)
        elif selected_engine == "gTTS" and GTTS_AVAILABLE:
                # ตรวจสอบและแก้ไข languageCode สำหรับ gTTS
                current_lang_code = config.get('languageCode', 'th')
                if '-' in current_lang_code:
                    # ถ้าเจอโค้ดแบบ th-TH ให้แปลงเป็น th
                    simple_lang_code = current_lang_code.split('-')[0]
                    self.log_message_gui(f"Correcting language code for gTTS on {channel_id_str}: '{current_lang_code}' -> '{simple_lang_code}'", level="DEBUG")
                    config['languageCode'] = simple_lang_code
                self._setup_gtts_options(options_frame, channel_id_str, config)
            # === สิ้นสุดบล็อกที่นำมาแทนที่ ===
        else:
            # Show unavailable message
            unavailable_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
            unavailable_frame.pack(fill="x", padx=5, pady=5)
            if selected_engine.endswith("(Not Impl)"):
                message = f"{selected_engine} - Feature not implemented yet"
                color = "orange"
            else:
                message = f"{selected_engine} - Library not available"
                color = "red"
            ctk.CTkLabel(unavailable_frame, text=message, text_color=color).pack(side="left", padx=5)
        # Auto-save when engine changes
        if not self._suppress_auto_save:
            self.schedule_auto_save()
    except Exception as e:
        self.log_message_gui(f"Error updating channel options for {channel_id_str}: {e}", level="ERROR")
   def on_engine_change(self, channel_id_str, new_engine):
    """Handle engine change with auto-save"""
    old_engine = self.tts_channel_configs[channel_id_str].get('engine', '')
    if old_engine != new_engine:
        self.log_message_gui(f"{channel_id_str}: Engine changed from {old_engine} to {new_engine}", level="INFO")
        self.after(200, self.auto_save_settings)
   def update_channel_config_value(self, channel_id_str, key, value):
    """Update channel config value with auto-save"""
    if channel_id_str in self.tts_channel_configs:
        old_value = self.tts_channel_configs[channel_id_str].get(key)
        self.tts_channel_configs[channel_id_str][key] = value
        # Auto-save only if value actually changed
        if old_value != value:
            self.after(500, self.auto_save_settings)  # Delay to prevent too frequent saves
   def _setup_google_tts_options(self, options_frame, channel_id_str, config):
    """Setup Google Cloud TTS options with quota management and emotion controls"""
    options_frame.grid_columnconfigure(1, weight=1)
    options_frame.grid_columnconfigure(3, weight=0)
    options_frame.grid_columnconfigure(5, weight=0)
    # Initialize quota system if not already done
    if not self.quota_manager:
        self.initialize_quota_system()
    # Key file section
    ctk.CTkLabel(options_frame, text="Key File (.json):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
    key_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
    key_frame.grid(row=0, column=1, columnspan=5, sticky="ew")
    key_frame.grid_columnconfigure(0, weight=1)
    key_var = tk.StringVar(value=config.get('google_key_path', ''))
    key_var.trace('w', lambda *args: self.on_google_key_change(channel_id_str, key_var.get()))
    key_entry = ctk.CTkEntry(key_frame, textvariable=key_var, width=250)
    key_entry.grid(row=0, column=0, padx=(0,5), sticky="ew")
    ctk.CTkButton(key_frame, text="Browse...", width=70, 
                 command=lambda v=key_var, e=key_entry: self.select_channel_google_key(channel_id_str, v, e)).grid(row=0, column=1, padx=0)
    key_status_label = ctk.CTkLabel(key_frame, text="Key not checked", text_color="gray", width=100)
    key_status_label.grid(row=0, column=2, padx=5)
    ctk.CTkButton(key_frame, text="Validate", width=70, 
                 command=lambda p=key_var, s=key_status_label: validate_google_json_key(p.get(), s)).grid(row=0, column=3, padx=0)
    self.tts_channel_widgets[channel_id_str]['google_key_var'] = key_var
    self.tts_channel_widgets[channel_id_str]['google_key_entry'] = key_entry
    self.tts_channel_widgets[channel_id_str]['google_key_status'] = key_status_label
    # Voice selection section
    voice_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
    voice_frame.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(10,0))
    ctk.CTkLabel(voice_frame, text="Voice List File (.txt):").pack(side="left", padx=5)
    voice_file_entry = ctk.CTkEntry(voice_frame, textvariable=self.google_tts_voice_file, width=150)
    voice_file_entry.pack(side="left", padx=5)
    ctk.CTkButton(voice_frame, text="Browse Voices File...", width=140,
                  command=self.select_and_load_google_voices).pack(side="left", padx=5)
    ctk.CTkLabel(voice_frame, text="Select Voice:").pack(side="left", padx=(15, 5))
    voice_var = tk.StringVar(value=config.get('name', ''))
    voice_var.trace('w', lambda *args: self.on_google_voice_change(channel_id_str, voice_var.get()))
    voice_menu = SearchableComboBox(voice_frame, variable=voice_var, values=self.google_tts_voices, width=250)
    voice_menu.pack(side="left", padx=5)
    if not voice_var.get() and self.google_tts_voices:
        voice_var.set(self.google_tts_voices[0])
    elif voice_var.get() not in self.google_tts_voices and self.google_tts_voices:
        voice_var.set(self.google_tts_voices[0])
    self.update_google_voice_params(channel_id_str, voice_var.get())
    self.tts_channel_widgets[channel_id_str]['google_voice_var'] = voice_var
    self.tts_channel_widgets[channel_id_str]['google_voice_menu'] = voice_menu
    # Quota Management Section
    quota_frame = ctk.CTkFrame(options_frame)
    quota_frame.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(10,5))
    quota_frame.grid_columnconfigure(1, weight=1)
    ctk.CTkLabel(quota_frame, text="Google TTS Quota Management:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="w")
    # สร้าง quota display widgets
    self.create_quota_display_widgets(quota_frame, channel_id_str)
    # Paid Feature Checkboxes
    paid_features_frame = ctk.CTkFrame(quota_frame, fg_color="transparent")
    paid_features_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=5)
    self.create_paid_feature_checkboxes(paid_features_frame, channel_id_str)
    # Audio properties section - compact version
    props_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
    props_frame.grid(row=3, column=0, columnspan=6, sticky="ew", pady=2)
    props_frame.grid_columnconfigure(1, weight=1)
    props_frame.grid_columnconfigure(3, weight=1)
    props_frame.grid_columnconfigure(5, weight=1)
    # Rate slider - แถวแรก
    ctk.CTkLabel(props_frame, text="Rate:", font=ctk.CTkFont(size=13)).grid(row=0, column=0, padx=5, pady=2, sticky="w")
    rate_var = tk.DoubleVar(value=config.get('speakingRate', 1.0))
    rate_slider = ctk.CTkSlider(props_frame, from_=0.25, to=4.0, number_of_steps=375, variable=rate_var, 
                          width=150, height=16)
    rate_slider.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
    rate_label = ctk.CTkLabel(props_frame, text=f"{rate_var.get():.2f}x", width=35, font=ctk.CTkFont(size=12))
    rate_label.grid(row=0, column=2, padx=2, pady=2)
    # Pitch slider - แถวเดียวกัน
    ctk.CTkLabel(props_frame, text="Pitch:", font=ctk.CTkFont(size=13)).grid(row=0, column=3, padx=5, pady=2, sticky="w")
    pitch_var = tk.DoubleVar(value=config.get('pitch', 0.0))
    pitch_slider = ctk.CTkSlider(props_frame, from_=-20.0, to=20.0, number_of_steps=400, variable=pitch_var, 
                           width=150, height=16)
    pitch_slider.grid(row=0, column=4, padx=5, pady=2, sticky="ew")
    pitch_label = ctk.CTkLabel(props_frame, text=f"{pitch_var.get():.1f}", width=35, font=ctk.CTkFont(size=10))
    pitch_label.grid(row=0, column=5, padx=2, pady=2)
    # Volume slider - แถวที่สอง
    ctk.CTkLabel(props_frame, text="Volume:", font=ctk.CTkFont(size=13)).grid(row=1, column=0, padx=5, pady=2, sticky="w")
    volume_var = tk.DoubleVar(value=config.get('volumeGainDb', 0.0))
    volume_slider = ctk.CTkSlider(props_frame, from_=-96.0, to=16.0, number_of_steps=1120, variable=volume_var, 
                            width=150, height=16)
    volume_slider.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
    volume_label = ctk.CTkLabel(props_frame, text=f"{volume_var.get():.1f}dB", width=45, font=ctk.CTkFont(size=10))
    volume_label.grid(row=1, column=2, padx=2, pady=2)
    # เก็บ references และ bind events
    rate_slider.configure(command=lambda v, l=rate_label, cid=channel_id_str: self.update_slider_label(v, l, 'speakingRate', "{:.2f}x", cid))
    pitch_slider.configure(command=lambda v, l=pitch_label, cid=channel_id_str: self.update_slider_label(v, l, 'pitch', "{:.1f}", cid))
    volume_slider.configure(command=lambda v, l=volume_label, cid=channel_id_str: self.update_slider_label(v, l, 'volumeGainDb', "{:.1f}dB", cid))
    self.tts_channel_widgets[channel_id_str]['google_rate_var'] = rate_var
    self.tts_channel_widgets[channel_id_str]['google_rate_label'] = rate_label
    self.tts_channel_widgets[channel_id_str]['google_pitch_var'] = pitch_var
    self.tts_channel_widgets[channel_id_str]['google_pitch_label'] = pitch_label
    self.tts_channel_widgets[channel_id_str]['google_volume_var'] = volume_var
    self.tts_channel_widgets[channel_id_str]['google_volume_label'] = volume_label
    # Emotion control section (for manual override when auto emotion is disabled)
    emotion_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
    emotion_frame.grid(row=4, column=0, columnspan=6, sticky="ew", pady=5)
    ctk.CTkLabel(emotion_frame, text="Manual Emotion Style:").pack(side="left", padx=5)
    emotions = ["neutral", "sad", "excited", "angry", "calm", "whisper", "custom"]
    emotion_var = tk.StringVar(value=config.get('emotion_style', 'neutral'))
    emotion_var.trace('w', lambda *args: self.on_manual_emotion_change(channel_id_str, emotion_var.get()))
    emotion_menu = ctk.CTkOptionMenu(
        emotion_frame, 
        variable=emotion_var, 
        values=emotions,
        command=lambda v, cid=channel_id_str: self.update_emotion_style(cid, v)
    )
    emotion_menu.pack(side="left", padx=5)
    # Custom SSML input
    ctk.CTkLabel(emotion_frame, text="Custom SSML:").pack(side="left", padx=(15,5))
    custom_ssml_entry = ctk.CTkEntry(emotion_frame, width=300, placeholder_text="<prosody rate='1.0'>{text}</prosody>")
    custom_ssml_entry.pack(side="left", padx=5)
    custom_ssml_entry.insert(0, config.get('custom_ssml', ''))
    custom_ssml_entry.bind('<KeyRelease>', lambda e, cid=channel_id_str: self.on_custom_ssml_change(cid, e.widget.get()))
    # Note about auto emotion
    note_label = ctk.CTkLabel(emotion_frame, text="หมายเหตุ: การตั้งค่านี้จะถูกใช้เมื่อปิดระบบ SSML อัตโนมัติ", 
                             text_color="gray", font=ctk.CTkFont(size=10))
    note_label.pack(side="right", padx=5)
    # เก็บ emotion widgets
    self.tts_channel_widgets[channel_id_str]['emotion_var'] = emotion_var
    self.tts_channel_widgets[channel_id_str]['custom_ssml_entry'] = custom_ssml_entry
    # Update emotion style on load
    self.update_emotion_style(channel_id_str, emotion_var.get())
    # Initial quota update
    if self.quota_manager:
        voice_name = voice_var.get()
        if voice_name:
            self.update_single_channel_quota_display(channel_id_str, voice_name)
    if not self.google_tts_voices:
        self.load_google_voices()
        voice_menu.set_values(self.google_tts_voices)
        if voice_var.get() not in self.google_tts_voices and self.google_tts_voices:
            voice_var.set(self.google_tts_voices[0])
            self.update_google_voice_params(channel_id_str, voice_var.get())
   def _setup_gtts_options(self, options_frame, channel_id_str, config):
    """Setup gTTS options with auto-save"""
    options_frame.grid_columnconfigure(1, weight=1)
    options_frame.grid_columnconfigure(3, weight=1)
    # Language selection
    ctk.CTkLabel(options_frame, text="Language:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
    lang_var = tk.StringVar(value=config.get('languageCode', 'th'))
    lang_var.trace('w', lambda *args: self.on_gtts_lang_change(channel_id_str, lang_var.get()))
    lang_codes = sorted(LANGUAGES.keys())
    lang_menu = SearchableComboBox(options_frame, variable=lang_var, values=lang_codes, width=100)
    lang_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")
    self.tts_channel_widgets[channel_id_str]['gtts_lang_var'] = lang_var
    # Speed control
    ctk.CTkLabel(options_frame, text="Speed:").grid(row=0, column=2, padx=(15,5), pady=5, sticky="e")
    speed_var = tk.DoubleVar(value=config.get('speed', 1.0))
    speed_slider = ctk.CTkSlider(options_frame, from_=0.5, to=2.0, number_of_steps=150, variable=speed_var)
    speed_slider.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
    speed_label = ctk.CTkLabel(options_frame, text=f"{speed_var.get():.1f}x", width=40)
    speed_label.grid(row=0, column=4, padx=5, pady=5)
    speed_slider.configure(command=lambda v, l=speed_label, cid=channel_id_str: self.update_slider_label(v, l, 'speed', "{:.1f}x", cid))
    self.tts_channel_widgets[channel_id_str]['gtts_speed_var'] = speed_var
    self.tts_channel_widgets[channel_id_str]['gtts_speed_label'] = speed_label
    # Pitch control
    ctk.CTkLabel(options_frame, text="Pitch (semitones):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
    pitch_var = tk.DoubleVar(value=config.get('pitch', 0.0))
    pitch_slider = ctk.CTkSlider(options_frame, from_=-12.0, to=12.0, number_of_steps=240, variable=pitch_var)
    pitch_slider.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
    pitch_label = ctk.CTkLabel(options_frame, text=f"{pitch_var.get():.1f} st", width=50)
    pitch_label.grid(row=1, column=4, padx=5, pady=5)
    pitch_slider.configure(command=lambda v, l=pitch_label, cid=channel_id_str: self.update_slider_label(v, l, 'pitch', "{:.1f} st", cid))
    self.tts_channel_widgets[channel_id_str]['gtts_pitch_var'] = pitch_var
    self.tts_channel_widgets[channel_id_str]['gtts_pitch_label'] = pitch_label
    # Request delay control
    delay_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
    delay_frame.grid(row=2, column=0, columnspan=5, padx=5, pady=5, sticky="ew")
    delay_frame.grid_columnconfigure(1, weight=1)
    ctk.CTkLabel(delay_frame, text="Request Delay (seconds):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
    delay_var = tk.DoubleVar(value=config.get('request_delay', 20.0))
    delay_slider = ctk.CTkSlider(delay_frame, from_=0.0, to=120.0, number_of_steps=1200, variable=delay_var)
    delay_slider.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    delay_label = ctk.CTkLabel(delay_frame, text=f"{delay_var.get():.1f}s", width=50)
    delay_label.grid(row=0, column=2, padx=5, pady=5)
    delay_slider.configure(command=lambda v, l=delay_label, cid=channel_id_str: self.update_slider_label(v, l, 'request_delay', "{:.1f}s", cid))
    # เพิ่มคำแนะนำ
    info_label = ctk.CTkLabel(delay_frame, text="แนะนำ: 40วิ/คำขอ สำหรับจำนวนมาก | 0-10วิ/คำขอ สำหรับ <500 คำขอ (ป้องกัน Error 429)", 
                            text_color="orange", font=ctk.CTkFont(size=10))
    info_label.grid(row=1, column=0, columnspan=3, padx=5, pady=2, sticky="w")
    self.tts_channel_widgets[channel_id_str]['gtts_delay_var'] = delay_var
    self.tts_channel_widgets[channel_id_str]['gtts_delay_label'] = delay_label
   def on_google_key_change(self, channel_id_str, new_key_path):
    """Handle Google key change with auto-save"""
    self.update_channel_config_value(channel_id_str, 'google_key_path', new_key_path)
   def on_google_voice_change(self, channel_id_str, new_voice):
        """Handle Google voice change with auto-save and correct quota display update"""
        self.update_google_voice_params(channel_id_str, new_voice)
        # Update quota display when voice changes
        if self.quota_manager and new_voice:
            # ดึง key_path ของ channel นี้
            key_path = self.tts_channel_configs[channel_id_str].get('google_key_path')
            voice_name = self.tts_channel_configs[channel_id_str].get('name')
            if key_path and voice_name:
                self.update_single_channel_quota_display(channel_id_str, voice_name)
    # === เพิ่มฟังก์ชันใหม่สำหรับแสดงสรุป Quota ===
   def show_full_quota_summary(self):
        """แสดงสรุปการใช้งาน Quota ของทุก API Key ใน Toplevel window"""
        if not self.quota_manager:
            messagebox.showinfo("Quota Info", "Quota management system is not initialized. Please set an output folder.")
            return
        summary_text = get_google_tts_usage_summary(self.output_folder.get())
        dialog = ctk.CTkToplevel(self)
        dialog.title("Google TTS Quota Summary (All Keys)")
        dialog.geometry("800x600")
        dialog.transient(self)
        dialog.grab_set()
        text_widget = ctk.CTkTextbox(dialog, wrap="word", font=(FONT_NAME, FONT_SIZE-2))
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        text_widget.insert("1.0", summary_text)
        text_widget.configure(state="disabled")
        close_button = ctk.CTkButton(dialog, text="Close", command=dialog.destroy)
        close_button.pack(pady=10)
        self.log_message_gui("Displayed full quota summary.", level="INFO")
   def on_manual_emotion_change(self, channel_id_str, new_emotion):
    """Handle manual emotion change with auto-save"""
    self.update_emotion_style(channel_id_str, new_emotion)
   def on_custom_ssml_change(self, channel_id_str, ssml_text):
    """Handle custom SSML change with auto-save"""
    self.update_custom_ssml(channel_id_str, ssml_text)
   def on_gtts_lang_change(self, channel_id_str, new_lang):
    """Handle gTTS language change with auto-save"""
    self.update_channel_config_value(channel_id_str, 'languageCode', new_lang)
   def update_emotion_style(self, channel_id_str, emotion):
    """อัพเดทรูปแบบอารมณ์สำหรับ channel with auto-save"""
    self.update_channel_config_value(channel_id_str, 'emotion_style', emotion)
    # ถ้าเป็น custom ให้แสดง custom SSML input
    custom_entry = self.tts_channel_widgets[channel_id_str].get('custom_ssml_entry')
    if custom_entry:
        if emotion == "custom":
            custom_entry.configure(state="normal")
        else:
            custom_entry.configure(state="disabled")
   def update_custom_ssml(self, channel_id_str, ssml_text):
    """อัพเดท custom SSML with auto-save"""
    self.update_channel_config_value(channel_id_str, 'custom_ssml', ssml_text)
   def select_channel_google_key(self, channel_id_str, key_var, key_entry_widget):
    """Select Google key for channel with auto-save"""
    file_selected = filedialog.askopenfilename(
        title=f"Select Google Cloud Key for {channel_id_str}", 
        filetypes=(("JSON files", "*.json"),)
    )
    if file_selected:
        key_var.set(file_selected)
        key_entry_widget.delete(0, tk.END)
        key_entry_widget.insert(0, file_selected)
        self.update_channel_config_value(channel_id_str, 'google_key_path', file_selected)
       # Validate key
        status_widget = self.tts_channel_widgets[channel_id_str].get('google_key_status')
        if status_widget:
           validate_google_json_key(file_selected, status_widget)
       # Auto-save immediately
        self.auto_save_settings()
        self.log_message_gui(f"{channel_id_str}: Selected Google key {os.path.basename(file_selected)}", level="INFO")
   def update_slider_label(self, value, label_widget, config_key, format_str, channel_id_str):
    """Update slider label and save config with auto-save"""
    label_widget.configure(text=format_str.format(value))
    self.update_channel_config_value(channel_id_str, config_key, value)
   def select_and_load_google_voices(self):
    """Opens a file dialog to select the Google voices file (.txt) and then loads it with auto-save"""
    initial_dir = os.path.dirname(self.google_tts_voice_file.get()) if os.path.exists(self.google_tts_voice_file.get()) else "."
    file_path = filedialog.askopenfilename(
       title="เลือกไฟล์ Voice List (.txt)",
       filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
       initialdir=initial_dir
      )
    if file_path:
       self.google_tts_voice_file.set(file_path)
       # 1. โหลดข้อมูลเสียงเข้ามาเก็บในหน่วยความจำก่อน
       self.load_google_voices()
       # 2. จากนั้น สั่งให้ UI ของทุกช่องที่เกี่ยวข้องวาดตัวเองใหม่ทั้งหมด
       self.refresh_all_google_tts_channels()
       # 3. Auto-save หลังโหลดเสียง
       self.auto_save_settings()
       self.log_message_gui(f"Loaded Google voices from: {os.path.basename(file_path)}", level="INFO")
    else:
       self.log_message_gui("Voice file selection cancelled.", level="INFO")
   def load_google_voices(self):
    """Load Google voices with error handling"""
    voice_file_path = self.google_tts_voice_file.get()
    if not voice_file_path or not os.path.exists(voice_file_path):
       self.log_message_gui(f"Voice file not found or not selected: {voice_file_path}", level="WARNING")
       self.google_tts_voices = []
       self.google_voice_details = {}
       return
    try:
       voices = []
       self.google_voice_details = {}
       with open(voice_file_path, "r", encoding="utf-8") as f:
           for line in f:
               parts = line.strip().split('\t')
               if len(parts) >= 5:
                   country, quality, lang_code, voice_name, gender = parts[0], parts[1], parts[2], parts[3], parts[4]
                   display_name = f"{voice_name} ({gender}, {lang_code})"
                   voices.append(display_name)
                   self.google_voice_details[display_name] = {
                       'name': voice_name, 
                       'languageCode': lang_code, 
                       'ssmlGender': gender
                   }
       self.google_tts_voices = sorted(voices)
       self.log_message_gui(f"Loaded {len(self.google_tts_voices)} Google TTS voices from {voice_file_path}", level="INFO")
    except Exception as e:
       messagebox.showerror("Load Voices Error", f"Error reading voice file:\n{e}")
       self.log_message_gui(f"Error loading Google voices from {voice_file_path}: {e}", level="ERROR")
   def refresh_all_google_tts_channels(self):
    """
    บังคับให้ช่อง Google Cloud TTS ที่มีอยู่ทั้งหมดทำการวาด UI ใหม่
    เพื่อให้รายการเสียงใน dropdown อัปเดตหลังจากโหลดไฟล์ใหม่
    """
    self.log_message_gui("Refreshing Google TTS channel UI...", level="DEBUG")
    # ทำซ้ำบนสำเนาของ keys เพื่อป้องกันปัญหาขณะแก้ไข dictionary
    for cid in list(self.tts_channel_widgets.keys()):
        widgets = self.tts_channel_widgets.get(cid)
        # ตรวจสอบว่าช่องนี้เป็น Google TTS หรือไม่
        if widgets and widgets.get('engine_var', tk.StringVar()).get() == "Google Cloud TTS":
            # คำสั่งนี้จะลบและสร้าง UI ส่วน options ของช่องนั้นๆ ใหม่ทั้งหมด
            self.update_channel_options(cid, 'Google Cloud TTS')
   def update_google_voice_params(self, channel_id_str, selected_display_name):
      """Update Google voice parameters with auto-save"""
      if channel_id_str in self.tts_channel_configs and selected_display_name in self.google_voice_details:
       details = self.google_voice_details[selected_display_name]
       self.update_channel_config_value(channel_id_str, 'name', details['name'])
       self.update_channel_config_value(channel_id_str, 'languageCode', details['languageCode'])
       self.update_channel_config_value(channel_id_str, 'ssmlGender', details['ssmlGender'])
       # อัพเดท quota display เมื่อเปลี่ยนเสียง
       if self.quota_manager:
           self.update_single_channel_quota_display(channel_id_str, details['name'])
      elif channel_id_str in self.tts_channel_configs:
       self.update_channel_config_value(channel_id_str, 'name', selected_display_name)
       self.update_channel_config_value(channel_id_str, 'languageCode', None)
       self.update_channel_config_value(channel_id_str, 'ssmlGender', None)
       # อัพเดท quota display
       if self.quota_manager and selected_display_name:
           self.update_single_channel_quota_display(channel_id_str, selected_display_name)
   def test_selected_voice(self, channel_id_str):
    """Test voice with comprehensive validation and error handling"""
    try:
        # Validate output folder
        if not self.output_folder.get() or not os.path.isdir(self.output_folder.get()):
            messagebox.showerror("Error", "Output folder must be set and valid before testing voice.")
            return
        # Validate channel exists
        if channel_id_str not in self.tts_channel_configs:
            messagebox.showerror("Error", f"Configuration for {channel_id_str} not found.")
            return
        config = self.tts_channel_configs[channel_id_str]
        output_format = getattr(self, 'tts_output_format_var', tk.StringVar(value='wav')).get()
        # Validate configuration before testing
        engine = config.get('engine')
        if engine == "Google Cloud TTS":
            key_path = config.get('google_key_path')
            if not key_path or not os.path.exists(key_path):
                messagebox.showerror("Configuration Error", 
                                   f"{channel_id_str}: Google TTS key file not found.\n\n"
                                   f"Please select a valid key file before testing.")
                return
            voice_name = config.get('name')
            if not voice_name:
                messagebox.showerror("Configuration Error", 
                                   f"{channel_id_str}: No voice selected.\n\n"
                                   f"Please select a voice before testing.")
                return
        elif engine == "gTTS":
            lang_code = config.get('languageCode')
            if not lang_code:
                messagebox.showerror("Configuration Error", 
                                   f"{channel_id_str}: No language code set.\n\n"
                                   f"Please select a language before testing.")
                return
        else:
            messagebox.showerror("Engine Error", 
                               f"{channel_id_str}: Engine '{engine}' is not supported or not available.")
            return
        # Get emotion system settings for test
        emotion_mode = getattr(self, 'emotion_mode_var', tk.StringVar(value='normal')).get()
        use_auto_emotion = emotion_mode in ['auto_simple', 'auto_advanced']
        use_advanced_emotion = emotion_mode == 'auto_advanced'
        emotion_analyzer = None
        ssml_generator = None
        if use_auto_emotion:
            try:
                emotion_analyzer = self.emotion_analyzer if hasattr(self, 'emotion_analyzer') and self.emotion_analyzer else None
                ssml_generator = self.ssml_generator if hasattr(self, 'ssml_generator') and self.ssml_generator else None
                if not emotion_analyzer:
                    if self.initialize_emotion_system():
                        emotion_analyzer = self.emotion_analyzer
                        ssml_generator = self.ssml_generator
                    else:
                        self.log_message_gui("Failed to initialize emotion system for test, using normal mode", level="WARNING")
                        use_auto_emotion = False
            except Exception as e:
                self.log_message_gui(f"Error with emotion system for test: {e}", level="WARNING")
                use_auto_emotion = False
        # Show confirmation with configuration summary
        engine_info = f"Engine: {engine}"
        if engine == "Google Cloud TTS":
            voice_name = config.get('name', 'Not selected')
            engine_info += f"\nVoice: {voice_name}"
        elif engine == "gTTS":
            lang_code = config.get('languageCode', 'Not selected')
            engine_info += f"\nLanguage: {lang_code}"
        emotion_info = ""
        if use_auto_emotion:
            mode_text = "Advanced (sentence splitting)" if use_advanced_emotion else "Simple (first keyword)"
            emotion_info = f"\nEmotion: {mode_text}"
        else:
            emotion_style = config.get('emotion_style', 'neutral')
            emotion_info = f"\nEmotion: Manual ({emotion_style})"
        result = messagebox.askyesno("Test Voice", 
                                   f"Test voice for {channel_id_str}?\n\n{engine_info}{emotion_info}\n\nOutput format: {output_format}")
        if not result:
            return
        # Run test in separate thread
        def run_test():
            try:
                test_channel_voice(config, self.output_folder.get(), output_format, self.log_text, 
                                 emotion_analyzer, ssml_generator, use_auto_emotion, use_advanced_emotion)
            except Exception as e:
                self.log_message_gui(f"Error in voice test thread: {e}", level="ERROR")
        test_thread = threading.Thread(target=run_test, daemon=True)
        test_thread.start()
        self.log_message_gui(f"Started voice test for {channel_id_str}", level="INFO")
    except Exception as e:
        error_msg = f"Error testing voice for {channel_id_str}: {e}"
        self.log_message_gui(error_msg, level="ERROR")
        messagebox.showerror("Test Voice Error", error_msg)
   def duplicate_tts_channel(self, source_channel_id):
      """Duplicate existing TTS channel with auto-save"""
      if source_channel_id not in self.tts_channel_configs:
       return
   # Copy source configuration
      source_config = self.tts_channel_configs[source_channel_id].copy()
   # Create new channel with copied config
      self.add_tts_channel(source_config)
      self.log_message_gui(f"Duplicated {source_channel_id}", level="INFO")
   def export_channel_config(self, channel_id_str):
      """Export channel configuration to file"""
      if channel_id_str not in self.tts_channel_configs:
       return
      config = self.tts_channel_configs[channel_id_str]
   # Select export file
      file_path = filedialog.asksaveasfilename(
       title=f"Export {channel_id_str} Configuration",
       filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
       defaultextension=".json",
       initialdir=self.output_folder.get()
      )
      if file_path:
       try:
           with open(file_path, 'w', encoding='utf-8') as f:
               json.dump(config, f, ensure_ascii=False, indent=2)
           messagebox.showinfo("Export Complete", f"{channel_id_str} configuration exported to:\n{file_path}")
           self.log_message_gui(f"Exported {channel_id_str} config to: {os.path.basename(file_path)}", level="INFO")
       except Exception as e:
           messagebox.showerror("Export Error", f"Failed to export configuration:\n{e}")
           self.log_message_gui(f"Error exporting {channel_id_str} config: {e}", level="ERROR")
   def import_channel_config(self):
      """Import channel configuration from file"""
      file_path = filedialog.askopenfilename(
       title="Import Channel Configuration",
       filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
       initialdir=self.output_folder.get()
      )
      if file_path:
       try:
           with open(file_path, 'r', encoding='utf-8') as f:
               config = json.load(f)
           # Validate configuration
           required_keys = ['engine']
           if not all(key in config for key in required_keys):
               messagebox.showerror("Import Error", "Invalid configuration file. Missing required keys.")
               return
           # Add channel with imported config
           self.add_tts_channel(config)
           messagebox.showinfo("Import Complete", f"Channel configuration imported from:\n{file_path}")
           self.log_message_gui(f"Imported channel config from: {os.path.basename(file_path)}", level="INFO")
       except Exception as e:
           messagebox.showerror("Import Error", f"Failed to import configuration:\n{e}")
           self.log_message_gui(f"Error importing channel config: {e}", level="ERROR")
   def validate_all_channels(self):
      """Validate all TTS channel configurations"""
      issues = []
      for channel_id, config in self.tts_channel_configs.items():
       channel_issues = self.validate_single_channel(channel_id, config)
       if channel_issues:
           issues.extend([f"{channel_id}: {issue}" for issue in channel_issues])
      if issues:
       issue_text = "\n".join([f"• {issue}" for issue in issues])
       messagebox.showwarning("Channel Validation Issues", f"Found configuration issues:\n\n{issue_text}")
       return False
      else:
       messagebox.showinfo("Channel Validation", "All channels are properly configured!")
       return True
   def validate_single_channel(self, channel_id, config):
      """Validate single channel configuration"""
      issues = []
      engine = config.get('engine')
      if engine == "Google Cloud TTS":
       key_path = config.get('google_key_path')
       if not key_path:
           issues.append("Missing Google TTS key file path")
       elif not os.path.exists(key_path):
           issues.append("Google TTS key file not found")
       voice_name = config.get('name')
       if not voice_name:
           issues.append("No voice selected")
      elif engine == "gTTS":
       lang_code = config.get('languageCode')
       if not lang_code:
           issues.append("No language code set")
      else:
       issues.append(f"Unsupported engine: {engine}")
      return issues
# ===============================================
   def handle_critical_error(self, error, context="Unknown"):
    """Handle critical errors with proper logging and user notification"""
    try:
        error_msg = f"Critical error in {context}: {str(error)}"
        self.log_message_gui(error_msg, level="CRITICAL")
        # Try to save current state
        try:
            self.save_settings()
        except:
            pass
        # Show error to user
        messagebox.showerror("Critical Error", 
                           f"A critical error occurred in {context}:\n\n{str(error)}\n\n"
                           f"Please check the log for more details.")
    except Exception as e:
        print(f"Error in error handler: {e}")
   def validate_ui_state(self):
    """Validate UI state and fix common issues"""
    try:
        # Check if essential widgets exist
        essential_widgets = ['input_folder', 'output_folder', 'tts_channel_configs']
        for widget_name in essential_widgets:
            if not hasattr(self, widget_name):
                self.log_message_gui(f"Missing essential widget: {widget_name}", level="ERROR")
                return False
        # Check if emotion_mode_var exists
        if not hasattr(self, 'emotion_mode_var'):
            self.emotion_mode_var = tk.StringVar(value="normal")
            self.log_message_gui("Created missing emotion_mode_var", level="WARNING")
        # Validate TTS channels
        invalid_channels = []
        for channel_id, config in self.tts_channel_configs.items():
            if not isinstance(config, dict) or 'engine' not in config:
                invalid_channels.append(channel_id)
        # Remove invalid channels
        for channel_id in invalid_channels:
            try:
                del self.tts_channel_configs[channel_id]
                if channel_id in self.tts_channel_widgets:
                    del self.tts_channel_widgets[channel_id]
                self.log_message_gui(f"Removed invalid channel: {channel_id}", level="WARNING")
            except:
                pass
        return True
    except Exception as e:
        self.log_message_gui(f"Error validating UI state: {e}", level="ERROR")
        return False
   def emergency_save_all(self):
    """Emergency save all critical data"""
    try:
        self.log_message_gui("Performing emergency save...", level="WARNING")
        # Save settings
        try:
            self.save_settings()
            self.log_message_gui("Emergency save: Settings saved", level="INFO")
        except Exception as e:
            self.log_message_gui(f"Emergency save: Failed to save settings: {e}", level="ERROR")
        # Save quota data
        if hasattr(self, 'quota_manager') and self.quota_manager:
            try:
                self.quota_manager.save_quota_data()
                self.log_message_gui("Emergency save: Quota data saved", level="INFO")
            except Exception as e:
                self.log_message_gui(f"Emergency save: Failed to save quota data: {e}", level="ERROR")
        # Save emotion config
        if hasattr(self, 'emotion_analyzer') and self.emotion_analyzer:
            try:
                self.emotion_analyzer.save_emotion_config()
                self.log_message_gui("Emergency save: Emotion config saved", level="INFO")
            except Exception as e:
                self.log_message_gui(f"Emergency save: Failed to save emotion config: {e}", level="ERROR")
        return True
    except Exception as e:
        self.log_message_gui(f"Critical error in emergency save: {e}", level="CRITICAL")
        return False
   def check_system_health(self):
    """Check system health and report issues"""
    try:
        health_report = {
            'ui_state': self.validate_ui_state(),
            'folders_accessible': True,
            'channels_valid': len(self.tts_channel_configs) >= 0,
            'emotion_system': hasattr(self, 'emotion_analyzer') and self.emotion_analyzer is not None,
            'quota_system': hasattr(self, 'quota_manager') and self.quota_manager is not None
        }
        # Check folder accessibility
        try:
            input_folder = self.input_folder.get()
            output_folder = self.output_folder.get()
            if input_folder and not os.path.exists(input_folder):
                health_report['folders_accessible'] = False
                self.log_message_gui(f"Input folder not accessible: {input_folder}", level="WARNING")
            if output_folder and not os.path.exists(output_folder):
                try:
                    os.makedirs(output_folder, exist_ok=True)
                    self.log_message_gui(f"Created output folder: {output_folder}", level="INFO")
                except:
                    health_report['folders_accessible'] = False
                    self.log_message_gui(f"Output folder not accessible: {output_folder}", level="WARNING")
        except Exception as e:
            health_report['folders_accessible'] = False
            self.log_message_gui(f"Error checking folders: {e}", level="ERROR")
        # Log health status
        healthy_systems = sum(health_report.values())
        total_systems = len(health_report)
        if healthy_systems == total_systems:
            self.log_message_gui("System health check: All systems healthy", level="INFO")
        else:
            self.log_message_gui(f"System health check: {healthy_systems}/{total_systems} systems healthy", level="WARNING")
        return health_report
    except Exception as e:
        self.log_message_gui(f"Error in system health check: {e}", level="ERROR")
        return {}
# ===============================================
# MAIN APPLICATION ENTRY POINT
# ===============================================
if __name__ == "__main__":
    try:
        # Setup logging
        import logging
        logging.basicConfig(level=logging.INFO)
        print(f"Starting {APP_NAME} v{APP_VERSION}")
        print("Initializing application...")
        # Check dependencies
        dependencies_ok = True
        if not WHISPER_AVAILABLE:
            print("Warning: Whisper library not available")
        if not GOOGLE_STT_AVAILABLE:
            print("Warning: Google Cloud Speech library not available")
        if not GEMINI_AVAILABLE:
            print("Warning: Gemini library not available")
        if not GOOGLE_TTS_AVAILABLE:
            print("Warning: Google Cloud Text-to-Speech library not available")
        if not GTTS_AVAILABLE:
            print("Warning: gTTS library not available")
        # Create and run application
        app = AudioProcessorApp()
        print("Application created successfully")
        print("Starting main loop...")
        app.mainloop()
    except Exception as e:
        print(f"Critical error starting application: {e}")
        import traceback
        traceback.print_exc()
        # Show error dialog if possible
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Application Error", f"Failed to start application:\n\n{e}")
        except:
            pass