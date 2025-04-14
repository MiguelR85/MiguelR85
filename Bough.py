import PyPDF2
import speech_recognition as sr
import pyttsx3
from PIL import Image
import pytesseract
import cv2
import subprocess
import time
from datetime import datetime
import threading
import random
import os
import re
import sympy as sp
import pyaudio
import wave
import json
import requests
import platform
import whisper
import numpy as np
import sounddevice as sd
import deepgram
import assemblyai
import nemo.collections.asr as nemo_asr
import coqui_stt

# Cross-platform detection
is_mobile = platform.system() in ['Android', 'iOS'] or 'mobile' in platform.platform().lower()

# Cloud Sync Configuration
CLOUD_SYNC_ENABLED = False
CLOUD_CONFIG = {
    'api_key': '',
    'cloud_url': 'https://api.example.com'
}

# Smart Home Integration
SMART_HOME_DEVICES = {
    'lights': {
        'bedroom': '192.168.1.100',
        'living_room': '192.168.1.101'
    },
    'thermostat': '192.168.1.102'
}

# Initialize enhanced components

# Enhanced audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 8192
DEVICE = None  # Default audio device

# Initialize TTS with better voice
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # More natural voice
engine.setProperty('rate', 170)  # Slightly faster speech
alarm_time = None
todo_list = []
reminders = {}
is_standby = False
last_spoken_text = None
context_memory = []
command_history = []
knowledge_base = {}
knowledge_file = "knowledge_base.json"

# Load knowledge base from file
try:
    with open(knowledge_file, 'r') as f:
        knowledge_base = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    knowledge_base = {}

def speak(text, priority="normal"):
    global last_spoken_text
    
    # Avoid repeating the same text
    if text == last_spoken_text and priority != "alert":
        return
    
    try:
        # JARVIS-style speech patterns
        text = (text.replace("I am", "I'm")
                  .replace("you are", "you're")
                  .replace("cannot", "can't")
                  .replace("do not", "don't")
                  .replace("it is", "it's")
                  .replace("there is", "there's"))
        
        # Add JARVIS-style prefixes
        if not text.startswith(("Sir", "Welcome back", "Initializing", "Running")):
            prefixes = ["", ""]  # Remove "Sir" prefix for Vicky
            text = random.choice(prefixes) + text
        
        # Set lowest possible male voice
        voices = engine.getProperty('voices')
        # Find the deepest male voice available
        lowest_pitch_voice = None
        lowest_pitch = 1.0  # Higher numbers are higher pitch
        
        for voice in voices:
            if "male" in voice.name.lower():
                if "jarvis" in voice.name.lower():
                    # Prefer JARVIS-style voice if available
                    engine.setProperty('voice', voice.id)
                    break
                if voice.id < lowest_pitch:
                    lowest_pitch = voice.id
                    lowest_pitch_voice = voice
        else:
            if lowest_pitch_voice:
                engine.setProperty('voice', lowest_pitch_voice.id)
            else:
                # Fallback to first voice if no male voices found
                engine.setProperty('voice', voices[0].id)
        
        # Optimize speech parameters for clarity
        engine.setProperty('rate', 150)  # Slower speech for clarity
        engine.setProperty('volume', 0.8)  # Slightly lower volume
        try:
            engine.setProperty('pitch', 0.3)  # Very low pitch
        except:
            pass  # Skip if pitch adjustment not supported
        
        # JARVIS-style delivery with precise pauses
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        for i, sentence in enumerate(sentences):
            print(f"VICK: {sentence}")
            engine.say(sentence)
            # Add dramatic pause after certain statements
            if i < len(sentences)-1 and any(w in sentences[i].lower() for w in ['warning', 'alert', 'important']):
                engine.runAndWait()
                time.sleep(0.5)
            else:
                engine.runAndWait()
                time.sleep(0.2)
                
        # Update the last spoken text
        last_spoken_text = text
        
    except Exception as e:
        print(f"Error in speech synthesis: {str(e)}")

# Function to listen to user commands
import whisper
import tempfile
import pyaudio
import wave

def listen():
    """Voice recognition using multiple services"""
    print("\nListening for command...")
    
    try:
        # Initialize all recognition services
        dg_client = deepgram.Deepgram(os.getenv('DEEPGRAM_API_KEY'))
        aai_client = assemblyai.Client(os.getenv('ASSEMBLYAI_API_KEY'))
        nemo_model = nemo_asr.models.EncDecCTCModel.from_pretrained("stt_en_conformer_ctc_large")
        coqui_model = coqui_stt.Model(os.path.join(coqui_stt.__path__[0], 'model.tflite'))
        
        # Record audio (enhanced settings)
        CHUNK = 2048  # Increased buffer size
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100  # Higher sample rate
        RECORD_SECONDS = 7  # Longer recording window
        
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wf = wave.open(f.name, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # Parallel transcription
            with ThreadPoolExecutor() as executor:
                # Submit all transcription tasks
                whisper_future = executor.submit(whisper.transcribe, f.name)
                dg_future = executor.submit(dg_client.transcription.prerecorded, {'buffer': open(f.name, 'rb')})
                aai_future = executor.submit(aai_client.transcribe, f.name)
                nemo_future = executor.submit(nemo_model.transcribe, [f.name])
                coqui_future = executor.submit(coqui_model.stt, f.name)

                # Get results with timeout
                try:
                    results = {
                        'whisper': whisper_future.result(timeout=5),
                        'deepgram': dg_future.result(timeout=5)['results']['channels'][0]['alternatives'][0]['transcript'],
                        'assemblyai': aai_future.result(timeout=5).text,
                        'nemo': nemo_future.result(timeout=5)[0],
                        'coqui': coqui_future.result(timeout=5)
                    }
                except Exception as e:
                    print(f"Transcription error: {e}")
                    return None

            # Voting mechanism - select most common result
            from collections import Counter
            valid_results = [v.strip().lower() for v in results.values() if v.strip()]
            if not valid_results:
                return None
                
            most_common = Counter(valid_results).most_common(1)[0][0]
            print(f"Consensus command: {most_common}")
            return most_common
            
    except Exception as e:
        print(f"Voice recognition error: {e}")
        return None

def respond(text):
    """Generate natural language responses"""
    responses = {
        "greeting": ["Hello there!", "Hi! How can I help?", "Greetings!"],
        "time": [f"The time is {datetime.now().strftime('%I:%M %p')}"],
        "date": [f"Today is {datetime.now().strftime('%B %d, %Y')}"],
        "default": ["I didn't quite catch that", "Could you repeat that?"]
    }
    
    # Simple intent detection
    if any(word in text for word in ['hi', 'hello', 'hey']):
        return random.choice(responses["greeting"])
    elif 'time' in text:
        return random.choice(responses["time"])
    elif 'date' in text:
        return random.choice(responses["date"])
    else:
        return random.choice(responses["default"])



# Function to open applications
def open_application(app_name):
    """Open any application by searching common install locations"""
    try:
        # Search common Windows program locations
        search_paths = [
            os.path.expandvars("%ProgramFiles%"),
            os.path.expandvars("%ProgramFiles(x86)%"),
            os.path.expandvars("%LOCALAPPDATA%\\Programs"),
            os.path.expandvars("%APPDATA%"),
            os.path.expandvars("%SystemRoot%\\System32")
        ]
        
        # Search for executable with matching name
        for root in search_paths:
            for dirpath, dirnames, filenames in os.walk(root):
                for filename in filenames:
                    if filename.lower().startswith(app_name.lower()):
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in ['.exe', '.bat', '.cmd', '.msi']:
                            full_path = os.path.join(dirpath, filename)
                            subprocess.Popen([full_path])
                            speak(f"Opening {filename}")
                            return True
        
        # If not found, try direct execution (may work for system apps)
        try:
            subprocess.Popen([app_name])
            speak(f"Attempting to open {app_name}")
            return True
        except:
            speak(f"Could not find application: {app_name}")
            return False
            
    except Exception as e:
        print(f"Error opening application: {e}")
        speak("Sorry, I encountered an error while trying to open that application")
        return False

# Function to set an alarm
def set_alarm(alarm_time_str):
    global alarm_time
    try:
        alarm_time = datetime.strptime(alarm_time_str, "%I:%M %p")
        speak(f"Alarm set for {alarm_time.strftime('%I:%M %p')}.")
        threading.Thread(target=monitor_alarm).start()
    except ValueError:
        speak("Sorry, I didn't understand the time. Please say something like 'set alarm for 7:30 AM'.")

# Function to monitor the alarm
def monitor_alarm():
    global alarm_time
    while alarm_time:
        if datetime.now().strftime("%I:%M %p") == alarm_time.strftime("%I:%M %p"):
            speak("Wake up! Your alarm is ringing.")
            alarm_time = None
        time.sleep(10)

# Function to manage tasks
def add_task(task):
    if task:
        todo_list.append(task)
        speak(f"Task added: {task}")
    else:
        speak("Please enter a task name.")

def show_tasks():
    if todo_list:
        speak("Here are your tasks:")
        for i, task in enumerate(todo_list, 1):
            speak(f"Task {i}: {task}")
    else:
        speak("Your to-do list is empty.")

# Function to set reminders
def set_reminder(task, reminder_time_str):
    try:
        reminder_time = datetime.strptime(reminder_time_str, "%I:%M %p")
        reminders[task] = reminder_time
        speak(f"Reminder set for {task} at {reminder_time.strftime('%I:%M %p')}.")
        threading.Thread(target=monitor_reminder, args=(task, reminder_time)).start()
    except ValueError:
        speak("Sorry, I didn't understand the time. Please say something like 'remind me to call John at 3:00 PM'.")

# Function to monitor reminders
def monitor_reminder(task, reminder_time):
    while task in reminders:
        if datetime.now().strftime("%I:%M %p") == reminder_time.strftime("%I:%M %p"):
            speak(f"Reminder: {task}")
            del reminders[task]
        time.sleep(10)

# Function to send email
def send_email(to, subject, body):
    speak("Email functionality is not available in offline mode.")

# Function to add calendar event
def add_calendar_event(summary, start_time, end_time):
    speak("Calendar functionality is not available in offline mode.")

# Function to tell a joke
def tell_joke():
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Did you hear about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them.",
        "Why don't skeletons fight each other? They don't have the guts."
    ]
    speak(random.choice(jokes))


def play_rock_paper_scissors():
    choices = ["rock", "paper", "scissors"]
    speak("Let's play Rock-Paper-Scissors! Say your choice.")
    user_choice = listen().lower()
    if user_choice not in choices:
        speak("Invalid choice. Please say rock, paper, or scissors.")
        return
    ai_choice = random.choice(choices)
    speak(f"I chose {ai_choice}.")
    if user_choice == ai_choice:
        speak("It's a tie!")
    elif (user_choice == "rock" and ai_choice == "scissors") or \
         (user_choice == "paper" and ai_choice == "rock") or \
         (user_choice == "scissors" and ai_choice == "paper"):
        speak("You win!")
    else:
        speak("I win!")

def generate_poetry():
    lines = [
        "Roses are red, violets are blue,",
        "The stars shine bright, just like you.",
        "In the quiet of the night,",
        "Dreams take flight, out of sight.",
        "Whispers of the wind, so soft and low,",
        "Tell tales of places we long to go."
    ]
    poem = "\n".join(random.sample(lines, 4))
    speak("Here's a poem for you:")
    speak(poem)

def generate_meme(top_text, bottom_text):
    speak("Meme generation is not available in offline mode.")

def ask_trivia():
    questions = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "options": ["London", "Berlin", "Madrid", "Paris"]
        },
        {
            "question": "How many continents are there?",
            "answer": "7",
            "options": ["5", "6", "7", "8"]
        }
    ]
    q = random.choice(questions)
    speak(f"Here's a trivia question: {q['question']}")
    speak(f"Options: {', '.join(q['options'])}")
    user_answer = listen().lower()
    if user_answer == q['answer'].lower():
        speak("Correct! Well done!")
    else:
        speak(f"Sorry, the correct answer is {q['answer']}.")


def calculator():
    """Enhanced scientific calculator with offline capabilities"""
    speak("What would you like to calculate? You can ask about:")
    speak("- Equations (solve 2x + 5 = 15)")
    speak("- Unit conversions (convert 5 feet to meters)")
    speak("- Scientific constants (speed of light)")
    speak("- Statistics (mean of 5,10,15)")
    speak("- Plotting (plot sin(x))")
    
    query = listen()
    if not query:
        return
        
    try:
        # Equation solving
        if "solve" in query:
            eq = query.replace("solve", "").strip()
            x = sp.symbols('x')
            solution = sp.solve(eq, x)
            speak(f"The solution is {solution}")
            
        # Unit conversions
        elif "convert" in query:
            parts = query.replace("convert", "").strip().split(" to ")
            if len(parts) == 2:
                value, from_unit = parts[0].split()
                to_unit = parts[1]
                converted = sp.convert_to(float(value)*sp.Unit(from_unit), sp.Unit(to_unit))
                speak(f"{value} {from_unit} is {converted} {to_unit}")
                
        # Scientific constants
        elif any(word in query for word in ["constant", "speed of light", "gravitational"]):
            constants = {
                "speed of light": "299,792,458 m/s",
                "gravitational constant": "6.67430 × 10⁻¹¹ m³⋅kg⁻¹⋅s⁻²",
                "plank constant": "6.62607015 × 10⁻³⁴ J⋅Hz⁻¹"
            }
            for name, value in constants.items():
                if name in query:
                    speak(f"{name} is {value}")
                    return
                    
        # Statistics
        elif "mean of" in query or "average of" in query:
            nums = [float(n) for n in re.findall(r"\d+", query)]
            if nums:
                speak(f"The mean is {sum(nums)/len(nums):.2f}")
                
        # Plotting
        elif "plot" in query:
            expr = query.replace("plot", "").strip()
            x = sp.symbols('x')
            plot = sp.plot(sp.sympify(expr), (x, -10, 10), show=False)
            plot.save('plot.png')
            speak("I've saved the plot as plot.png")
            
        # Fallback to original calculator
        else:
            # Date calculations
            if "days between" in query or "days from" in query:
                dates = re.findall(r"(\d{1,2}/\d{1,2}/\d{2,4})", query)
                if len(dates) == 2:
                    date1 = datetime.strptime(dates[0], "%m/%d/%Y")
                    date2 = datetime.strptime(dates[1], "%m/%d/%Y")
                    delta = abs((date2 - date1).days)
                    speak(f"There are {delta} days between {dates[0]} and {dates[1]}")
                    return

            # Standard math evaluation
            result = sp.sympify(query).evalf()
            speak(f"The result is {result}")
            
    except Exception as e:
        speak(f"Sorry, I couldn't perform that calculation. Error: {e}")

def wireless_earphone_button_pressed():
    """Check if wireless earphone button is pressed with proper implementation"""
    try:
        # This would interface with actual hardware/Bluetooth in a real implementation
        # For now simulate random button presses for testing
        return random.random() < 0.1  # 10% chance of button press
    except Exception as e:
        print(f"Error checking earphone button: {e}")
        return False


def record_audio_by_button():
    speak("Please press the Play/Pause button on your wireless earphone to start recording and press it again to stop recording.")
    while True:
        if wireless_earphone_button_pressed():
            speak("Recording started. Please press the Play/Pause button again to stop recording.")
            # Start recording audio
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 2
            RATE = 44100
            RECORD_SECONDS = 60
            WAVE_OUTPUT_FILENAME = "output.wav"
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
                if wireless_earphone_button_pressed():
                    break
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            speak("Recording stopped and saved as output.wav")
            break

def get_formula_one_updates():
    speak("Formula One updates are not available in offline mode.")


# Cache for search results
search_cache = {}
CACHE_FILE = "search_cache.json"

def load_search_cache():
    """Load search results cache from file"""
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_search_cache():
    """Save search results cache to file"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(search_cache, f)

def clear_search_cache():
    """Clear the search results cache"""
    global search_cache
    search_cache = {}
    save_search_cache()
    return "Search cache cleared"

from concurrent.futures import ThreadPoolExecutor
import hashlib

def search_offline_content(query, file_types=None):
    """Search for information in local files with enhanced capabilities"""
    global search_cache
    
    # Create cache key with query and file_types
    cache_key = hashlib.md5((query + str(file_types)).encode()).hexdigest()
    
    # Check cache first
    if cache_key in search_cache:
        speak("Returning cached results...")
        return search_cache[cache_key]
        
    # Get all available drives on Windows
    if platform.system() == 'Windows':
        import string
        from ctypes import windll
        drives = []
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1:
                drives.append(f"{letter}:\\")
            bitmask >>= 1
    else:  # For Linux/Mac/Android
        drives = ['/']
    
    # Get system folders to exclude
    exclude_dirs = [
        'C:\\Windows', 'C:\\Program Files', 'C:\\Program Files (x86)',
        'C:\\ProgramData', '/system', '/proc', '/Library', '/System'
    ]
    
    # Supported file types with display names
    supported_extensions = {
        'Text': ['.txt', '.md', '.csv', '.json', '.xml'],
        'Office': ['.docx', '.xlsx', '.pptx'],
        'Code': ['.py', '.js', '.java', '.c', '.cpp', '.html', '.css'],
        'eBooks': ['.epub', '.mobi'],
        'Archives': ['.zip', '.rar', '.7z'],
        'Audio': ['.mp3', '.wav', '.ogg'],
        'Databases': ['.db', '.sqlite', '.mdb']
    }

    # Filter by requested file types if specified
    if file_types:
        file_types = [ft.lower() for ft in file_types]
        supported_extensions = {k:v for k,v in supported_extensions.items() 
                              if k.lower() in file_types}
    
    # Initialize results with all supported categories
    results = {category: [] for category in supported_extensions}
    results.update({
        'PDF': [],
        'Image': [],
        'Video': []
    })
    
    total_files = 0
    last_progress_time = time.time()
    
    # Search across all drives with enhanced feedback
    search_start = time.time()
    for drive_idx, drive in enumerate(drives):
        drive_start = time.time()
        speak(f"Searching drive {drive} ({drive_idx+1}/{len(drives)})...")
        
        for root, dirs, files in os.walk(drive):
            # Skip excluded directories
            if any(exclude_dir.lower() in root.lower() for exclude_dir in exclude_dirs):
                dirs[:] = []
                continue
                
            # Progress feedback
            if time.time() - last_progress_time > 30:
                elapsed = int(time.time() - search_start)
                speak(f"Searching... {elapsed//60}m {elapsed%60}s elapsed. Found {total_files} files with {sum(len(r) for r in results.values())} matches")
                last_progress_time = time.time()
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for file in files:
                    total_files += 1
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    futures.append(executor.submit(process_file, file_path, file_ext, query, results))
                
                # Wait for all futures to complete
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error processing file: {e}")
                
                # PDF files (now under 'PDF' category)
                if file_ext == '.pdf' and 'PDF' in results:
                    try:
                        with open(file_path, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            for i, page in enumerate(reader.pages):
                                text = page.extract_text()
                                if text and query.lower() in text.lower():
                                    results['pdf'].append({
                                        'file': file_path,
                                        'page': i+1,
                                        'text': text[:500] + '...'
                                    })
                                    if len(results['pdf']) >= 3:
                                        break
                    except Exception as e:
                        print(f"Error reading PDF {file_path}: {e}")

                # Image files (now under 'Image' category)
                elif file_ext in ('.png', '.jpg', '.jpeg') and 'Image' in results:
                    try:
                        img = Image.open(file_path)
                        text = pytesseract.image_to_string(img)
                        if query.lower() in text.lower():
                            results['image'].append({
                                'file': file_path,
                                'text': text[:500] + '...'
                            })
                            if len(results['image']) >= 3:
                                break
                    except Exception as e:
                        print(f"Error reading image {file_path}: {e}")

                # Video files (now under 'Video' category)  
                # Text files (now under 'Text' category)
                elif 'Text' in results and file_ext in supported_extensions['Text']:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            if query.lower() in text.lower():
                                results['text'].append({
                                    'file': file_path,
                                    'text': text[:500] + '...'
                                })
                    except Exception as e:
                        print(f"Error reading text file {file_path}: {e}")
                        
                # Office documents (now under 'Office' category)
                elif 'Office' in results and file_ext in supported_extensions['Office']:
                    try:
                        text = ''
                        if file_ext == '.docx':
                            try:
                                import docx
                                doc = docx.Document(file_path)
                                text = '\n'.join([para.text for para in doc.paragraphs])
                            except ImportError:
                                speak("python-docx module not installed - skipping Word documents")
                                supported_extensions['Office'].remove('.docx')
                                continue
                                
                        elif file_ext == '.xlsx':
                            try:
                                import openpyxl
                                wb = openpyxl.load_workbook(file_path)
                                text = ''
                                for sheet in wb.sheetnames:
                                    text += f"\nSheet: {sheet}\n"
                                    for row in wb[sheet].iter_rows(values_only=True):
                                        text += ' '.join(str(cell) for cell in row if cell) + '\n'
                            except ImportError:
                                speak("openpyxl module not installed - skipping Excel documents")
                                supported_extensions['Office'].remove('.xlsx')
                                continue
                                
                        elif file_ext == '.pptx':
                            try:
                                from pptx import Presentation
                                prs = Presentation(file_path)
                                text = ''
                                for slide in prs.slides:
                                    for shape in slide.shapes:
                                        if hasattr(shape, "text"):
                                            text += shape.text + '\n'
                            except ImportError:
                                speak("python-pptx module not installed - skipping PowerPoint documents")
                                supported_extensions['Office'].remove('.pptx')
                                continue
                                
                        if text and query.lower() in text.lower():
                            results['office'].append({
                                'file': file_path,
                                'text': text[:500] + '...'
                            })
                    except Exception as e:
                        print(f"Error reading office file {file_path}: {e}")
                        
                # Code files (now under 'Code' category)
                elif 'Code' in results and file_ext in supported_extensions['Code']:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            if query.lower() in text.lower():
                                results['code'].append({
                                    'file': file_path,
                                    'text': text[:500] + '...'
                                })
                    except Exception as e:
                        print(f"Error reading code file {file_path}: {e}")

                # eBook files
                elif 'eBooks' in results and file_ext in supported_extensions['eBooks']:
                    try:
                        if file_ext == '.epub':
                            import epub
                            book = epub.open_epub(file_path)
                            text = ''
                            for item in book.get_items():
                                if item.get_type() == epub.ITEM_DOCUMENT:
                                    text += item.get_content().decode('utf-8') + '\n'
                            book.close()
                            
                        if query.lower() in text.lower():
                            results['eBooks'].append({
                                'file': file_path,
                                'text': text[:500] + '...'
                            })
                    except Exception as e:
                        print(f"Error reading eBook {file_path}: {e}")

                # Archive files
                elif 'Archives' in results and file_ext in supported_extensions['Archives']:
                    try:
                        import zipfile
                        with zipfile.ZipFile(file_path) as z:
                            for name in z.namelist():
                                if name.lower().endswith(('.txt', '.md', '.csv')):
                                    with z.open(name) as f:
                                        text = f.read().decode('utf-8')
                                        if query.lower() in text.lower():
                                            results['Archives'].append({
                                                'file': f"{file_path}/{name}",
                                                'text': text[:500] + '...'
                                            })
                                            break
                    except Exception as e:
                        print(f"Error reading archive {file_path}: {e}")

    # Format results with improved presentation
    response = []
    found_results = False
    
    for category in sorted(results.keys()):
        if results[category]:
            found_results = True
            category_results = []
            for result in results[category][:5]:  # Increased to 5 results per category
                if 'page' in result:
                    entry = f"- {os.path.basename(result['file'])} (page {result['page']}):\n  {result['text']}"
                else:
                    entry = f"- {os.path.basename(result['file'])}:\n  {result['text']}"
                category_results.append(entry)
            
            if category_results:
                response.append(f"=== {category.upper()} ===")
                response.extend(category_results)
    
    if not found_results:
        final_response = "No results found in your files."
    else:
        final_response = "Search Results:\n\n" + "\n\n".join(response)
        # Cache the results (unless filtered by file type)
        if not file_types:
            search_cache[query] = final_response
            save_search_cache()
    
    return response

def answer_question():
    speak("What would you like to search for in your files?")
    query = listen()
    if not query:
        return
    
    speak("Searching your local files...")
    result = search_offline_content(query)
    speak(result)

# Note-taking feature
def take_notes():
    speak("Please say what you want to note.")
    note = listen()
    with open("notes.txt", "a") as f:
        f.write(note + "\n")
    speak("Note taken.")

# Text-to-speech feature
def text_to_speech():
    speak("Please say what you want to read.")
    text = listen()
    engine.say(text)
    engine.runAndWait()

# Translation feature
def translate(text):
    # Simple translation example - would need proper translation API in production
    translations = {
        "hello": "hola",
        "goodbye": "adiós",
        "thank you": "gracias"
    }
    return translations.get(text.lower(), "Translation not available")

def translate_command():
    speak("Please say what you want to translate.")
    text = listen()
    translation = translate(text)
    speak(translation)


# Math problem-solving feature
def solve_math_problem(problem):
    try:
        return str(eval(problem))
    except:
        return "Could not solve the problem"

def solve_math_command():
    speak("Please say what math problem you want to solve.")
    problem = listen()
    solution = solve_math_problem(problem)
    speak(solution)


# Flashcard feature
def create_flashcard():
    speak("Please say what you want to put on the front of the flashcard.")
    front = listen()
    speak("Please say what you want to put on the back of the flashcard.")
    back = listen()
    with open("flashcards.txt", "a") as f:
        f.write(front + ":" + back + "\n")
    speak("Flashcard created.")

# Audio recording feature
def record_audio():
    speak("Please say what you want to record.")
    recording = listen()
    with open("recording.wav", "wb") as f:
        f.write(recording)
    speak("Recording saved.")

# Voice-to-text feature
def voice_to_text():
    speak("Please say what you want to write.")
    text = listen()
    with open("text.txt", "w") as f:
        f.write(text)
    speak("Text written.")

# Dictionary feature
def look_up_word(word):
    # Basic dictionary implementation
    dictionary = {
        "hello": "a greeting",
        "world": "the earth with all its countries and peoples",
        "python": "a high-level programming language"
    }
    return dictionary.get(word.lower(), "Definition not found")

def look_up_word_command():
    speak("Please say what word you want to look up.")
    word = listen()
    definition = look_up_word(word)
    speak(definition)


# Thesaurus feature
def look_up_synonyms(word):
    # Basic thesaurus implementation
    thesaurus = {
        "happy": ["joyful", "cheerful", "content"],
        "sad": ["unhappy", "depressed", "melancholy"]
    }
    return thesaurus.get(word.lower(), ["No synonyms found"])

def look_up_synonyms_command():
    speak("Please say what word you want synonyms for.")
    word = listen()
    synonyms = look_up_synonyms(word)
    speak(f"Synonyms for {word}: {', '.join(synonyms)}")


# Study guide feature
def create_study_guide():
    speak("Please say what you want to put in the study guide.")
    guide = listen()
    with open("study_guide.txt", "w") as f:
        f.write(guide)
    speak("Study guide created.")

def remember_fact(fact, category="general"):
    """Store a personal fact in the knowledge base"""
    if category not in knowledge_base:
        knowledge_base[category] = []
    knowledge_base[category].append(fact)
    with open(knowledge_file, 'w') as f:
        json.dump(knowledge_base, f)
    return f"I'll remember that: {fact}"

def recall_fact(topic):
    """Retrieve facts about a topic from the knowledge base"""
    results = []
    for category, facts in knowledge_base.items():
        if topic.lower() in category.lower():
            results.extend(facts)
        for fact in facts:
            if topic.lower() in fact.lower():
                results.append(fact)
    return results if results else "I don't know anything about that."

def interpret_command(command):
    """Interpret user command with enhanced system control"""
    if not command:
        return None

    try:
        command = command.lower().strip()
        
        # Basic command mapping
        command_map = {
            'open browser': ('open', 'chrome'),
            'open notepad': ('open', 'notepad'),
            'open calculator': ('open', 'calculator'),
            'open word': ('open', 'winword'),
            'open excel': ('open', 'excel')
        }
        
        # Check if command matches any predefined phrase
        for phrase, action in command_map.items():
            if phrase in command:
                return action
        
        # System control commands
        if any(word in command for word in ['shutdown', 'turn off', 'power off']):
            return 'system_shutdown'
        elif any(word in command for word in ['restart', 'reboot']):
            return 'system_restart'
        elif any(word in command for word in ['lock', 'lock screen']):
            return 'system_lock'
        elif any(word in command for word in ['sleep', 'hibernate']):
            return 'system_sleep'
        
        # File operations
        elif any(word in command for word in ['create file', 'make file', 'new file']):
            return 'create_file'
        elif any(word in command for word in ['delete file', 'remove file']):
            return 'delete_file'
        elif any(word in command for word in ['move file', 'rename file']):
            return 'move_file'
            
        # Process control
        elif any(word in command for word in ['kill', 'end process', 'stop process']):
            return 'kill_process'
        elif any(word in command for word in ['list processes', 'running processes']):
            return 'list_processes'
            
        # Application control
        elif any(word in command for word in ['open', 'start', 'launch', 'run', 'begin']):
            return 'open'
        elif any(word in command for word in ['close', 'exit', 'quit']):
            return 'close_app'
            
        # Original commands
        elif any(word in command for word in ['alarm', 'reminder', 'timer', 'alert']):
            return 'set_alarm'
        elif any(word in command for word in ['add', 'create', 'new', 'task', 'todo', 'item']):
            return 'add_task'
        elif any(word in command for word in ['show', 'list', 'tasks', 'todos', 'items']):
            return 'show_tasks'
        elif any(word in command for word in ['remind', 'notify', 'remember', 'alert']):
            return 'set_reminder'
        elif any(word in command for word in ['joke', 'funny', 'humor', 'laugh']):
            return 'tell_joke'
        elif any(word in command for word in ['time', 'clock', 'current time', 'what time'] ):
            return 'get_time'
        elif any(word in command for word in ['date', 'today', 'current date', 'what date']):
            return 'get_date'
        elif any(word in command for word in ['calculate', 'math', 'compute', 'solve']):
            return 'calculator'
        elif any(word in command for word in ['f1', 'formula one', 'racing', 'grand prix']):
            return 'formula_one_updates'
        elif any(word in command for word in ['sleep', 'pause', 'rest', 'idle']):
            return 'standby'
        elif any(word in command for word in ['wake', 'resume', 'start', 'hello']):
            return 'wake_up'
        elif any(word in command for word in ['help', 'assist', 'support', 'guide']):
            return 'help'
        else:
            return None
            
    except Exception as e:
        print(f"Command interpretation error: {e}")
        return None

def control_smart_home(device, action):
    """Control smart home devices with enhanced capabilities"""
    # Lights control
    if device in SMART_HOME_DEVICES.get('lights', {}):
        ip = SMART_HOME_DEVICES['lights'][device]
        try:
            requests.post(f"http://{ip}/control", 
                        json={'action': action},
                        timeout=3)
            speak(f"Turning {action} {device} lights")
            return True
        except Exception as e:
            print(f"Smart home error: {e}")
            speak(f"Failed to control {device} lights")
            return False
            
    # Thermostat control        
    elif device == 'thermostat' and (action.isdigit() or action in ['heat', 'cool', 'off']):
        ip = SMART_HOME_DEVICES['thermostat']
        try:
            if action.isdigit():
                requests.post(f"http://{ip}/set", 
                            json={'temp': int(action)},
                            timeout=3)
                speak(f"Setting thermostat to {action} degrees")
            else:
                requests.post(f"http://{ip}/mode", 
                            json={'mode': action},
                            timeout=3)
                speak(f"Setting thermostat mode to {action}")
            return True
        except Exception as e:
            print(f"Thermostat error: {e}")
            speak("Failed to adjust thermostat")
            return False
            
    # Security system control
    elif device == 'security' and action in ['arm', 'disarm']:
        if 'security' in SMART_HOME_DEVICES:
            ip = SMART_HOME_DEVICES['security']
            try:
                requests.post(f"http://{ip}/{action}",
                            timeout=3)
                speak(f"Security system {action}ed")
                return True
            except Exception as e:
                print(f"Security system error: {e}")
                speak(f"Failed to {action} security system")
                return False
                
    return False

def sync_with_cloud():
    """Sync data with cloud service"""
    if not CLOUD_SYNC_ENABLED:
        return False
    
    data = {
        'todos': todo_list,
        'reminders': reminders,
        'context': context_memory[-10:] if context_memory else []
    }
    
    try:
        response = requests.post(
            f"{CLOUD_CONFIG['cloud_url']}/sync",
            json=data,
            headers={'Authorization': f"Bearer {CLOUD_CONFIG['api_key']}"}
        )
        return response.status_code == 200
    except:
        return False

def main():
    global is_standby
    
    # Initial greeting
    speak("Initializing all systems. Online and ready.", "alert")
    
    while True:
        try:
            if not is_standby:
                # Skip initial greeting - just listen for commands
                
                command = listen()
                if command:
                    command_history.append(command)
                    result = interpret_command(command)
                    
                    if not result:
                        speak("I didn't quite catch that. Could you rephrase?")
                        continue

                    # Handle command execution
                    if command.startswith('open '):
                        app_name = command[5:]
                        if app_name:
                            open_application(app_name)
                        else:
                            speak("Please specify an application to open")

                    elif "alarm" in command or "wake me" in command:
                        alarm_time_str = command.split("alarm for ")[-1] if "alarm for" in command else command.split("wake me at ")[-1]
                        if alarm_time_str:
                            set_alarm(alarm_time_str)
                        else:
                            speak("Please specify a time for the alarm")

                    elif "add task" in command or "add " in command:
                        task = command.split("add task ")[-1] if "add task" in command else command.split("add ")[-1]
                        add_task(task)
                    
                    elif "show tasks" in command or "list tasks" in command:
                        show_tasks()
                    
                    elif "remind me to" in command:
                        reminder_parts = command.split("remind me to ")
                        if len(reminder_parts) > 1:
                            task = reminder_parts[1].split(" at ")[0]
                            reminder_time_str = reminder_parts[1].split(" at ")[1]
                            set_reminder(task, reminder_time_str)
                    
                    elif "joke" in command or "tell me something funny" in command:
                        tell_joke()
                    
                    elif "time" in command or "what time is it" in command:
                        speak("The current time is " + str(datetime.now().strftime("%H:%M:%S")))
                    
                    elif "date" in command or "what date is it" in command:
                        speak("The current date is " + str(datetime.now().strftime("%d/%m/%Y")))
                    
                    elif "calculate" in command or "math" in command:
                        calculator()
                    
                    elif "formula one" in command or "f1" in command:
                        get_formula_one_updates()
                    
                    elif "sleep" in command or "standby" in command:
                        is_standby = True
                        speak("Goodbye! I'll be here when you need me again.")
                    
                    elif "wake up" in command or "hello" in command:
                        is_standby = False
                        speak("Hello! I'm awake now.")
                    
                    elif command.startswith(('remember that', 'remember this')):
                        fact = command.split('that')[-1] if 'that' in command else command.split('this')[-1]
                        response = remember_fact(fact.strip())
                        speak(response)
                    elif command.startswith(('what do you know about', 'tell me about')):
                        topic = command.split('about')[-1].strip()
                        facts = recall_fact(topic)
                        if isinstance(facts, list):
                            speak(f"I know these things about {topic}:")
                            for fact in facts:
                                speak(fact)
                        else:
                            speak(facts)
                    elif "help" in command:
                        speak("Here are some things I can do:")
                        speak("- Open applications (say 'open browser')")
                        speak("- Set alarms and reminders (say 'set alarm for 7am')")
                        speak("- Manage tasks (say 'add task' or 'show tasks')")
                        speak("- Tell jokes (say 'tell me a joke')")
                        speak("- Perform calculations (say 'calculate 2 plus 2')")
                        speak("- Get time/date (say 'what time is it')")
        except Exception as e:
            print(f"Error in main loop: {e}")
            print("[Error occurred. Ready for next command]")

if __name__ == "__main__":
    main()