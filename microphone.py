import time
import platform
import wave

from sounddevice import InputStream
import numpy as np
from pynput import keyboard
import openai

import console

# Constants
SAMPLE_RATE = 44100
CHANNELS = 2
FILENAME = 'temp/microphone.wav'
MIN_RECORD_TIME = 0.5
REFRESH_RATE = 0.05
WAVE_LEN = 25
BLOCK_CHARS = '▁▂▃▄▅▆▇█'
SUBMIT_TEXT = True

# Global variables
transcribed_text = ''
waveform = []
buffer = []
total_frames = 0
pressed_keys = set()
chat = None

def block(val, low=0, hi=50):
    if hi == low:
        return BLOCK_CHARS[0]
    val = max(min(val, hi), low)
    index = (len(BLOCK_CHARS) - 1) * (val - low) / (hi - low)
    return BLOCK_CHARS[int(index)]

def resample(data, new_length):
    data = np.array(data)
    old_indices = np.arange(len(data))
    new_indices = np.linspace(0, len(data)-1, new_length)
    return np.interp(new_indices, old_indices, data)

def print_sound(first=False, last=False):
    global waveform, transcribed_text
    low = np.min(waveform)
    if len(waveform) > WAVE_LEN: # Resample to fit in 25 characters
        print_wave = resample(waveform, WAVE_LEN)
    else:
        print_wave = waveform + [low] * (WAVE_LEN-len(waveform))
    
    mean, std = np.mean(waveform), np.std(waveform)
    low, high = mean-(std*1.0), mean+(std*1.5)
    blocks = ''.join(block(v, low, high) for v in print_wave)
    prefix = '' if first else chat.user_prefix()
    length = float(total_frames)/SAMPLE_RATE

    light_mode = console.bright(console.terminal_background)
    tc = '#404040' if light_mode else '#FFFFFF'
    fg = console.blend(tc, console.terminal_background, 0.15)
    bg = console.blend(tc, console.terminal_background, 0.85)
    wave_string = f'{prefix}▕[{fg} on {bg}]{blocks}[/]▏[od.cyan_dim]{length:.2f}s[/]'

    if last:
        console.print(f'{wave_string}  {transcribed_text}', end='')
    else:
        console.print(wave_string, end='\r')

def callback(data, frames, time, status):
    global buffer, total_frames, waveform
    first_sample = (total_frames == 0)
    buffer.extend(data.copy())
    total_frames += frames

    # Estimate the magnitude (volume)
    average_abs = np.mean(np.abs(data))
    waveform.append(float(average_abs))

    print_sound(first=first_sample)

def on_press(key):
    global pressed_keys
    pressed_keys.add(key)

def on_release(key):
    global pressed_keys
    if key in pressed_keys:
        pressed_keys.remove(key)

def transcribe_audio(file):
    """Transcribes audio file using OpenAI's Whisper model"""
    client = openai.OpenAI()
    with open(file, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1", file=f
        )
    return response.text

def watch_record_audio(conversation=None, keys=None):
    global buffer, total_frames, transcribed_text, waveform, chat
    chat = conversation

    if keys == None:
        if platform.system() == 'Darwin': # Mac
            keys = (keyboard.Key.ctrl, keyboard.Key.shift)
        else: # Windows and Linux
            keys = (keyboard.Key.ctrl_l, keyboard.Key.alt_l)

    controller = keyboard.Controller()
    
    with keyboard.Listener(on_press=on_press, on_release=on_release, suppress=(platform.system() == 'Darwin')):
        while True:
            if all(k in pressed_keys for k in keys):
                buffer = [] # Clear the buffer
                waveform = []

                stream = InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback, blocksize=SAMPLE_RATE//20)
                stream.start()

                while all(k in pressed_keys for k in keys):
                    time.sleep(REFRESH_RATE) # Sleep until key combo released

                stream.stop()
                if float(total_frames)/SAMPLE_RATE > MIN_RECORD_TIME:
                    data = (np.array(buffer) * np.iinfo(np.int16).max).astype(np.int16)
                    with wave.open(FILENAME, 'wb') as file:
                        file.setnchannels(CHANNELS)  # Mono audio
                        file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                        file.setframerate(SAMPLE_RATE)
                        file.writeframes(data.tobytes())
                    transcribed_text = transcribe_audio(FILENAME)
                    print_sound(last=True)

                total_frames = 0

                if SUBMIT_TEXT:
                    controller.press(keyboard.Key.enter)
                    controller.release(keyboard.Key.enter)
            
            time.sleep(REFRESH_RATE)
