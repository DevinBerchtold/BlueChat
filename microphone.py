import time
import os
import platform

import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import resample
import numpy as np
from pynput import keyboard
import openai

import console

# Global variables
TEXT = ''
SUBMIT_TEXT = True
waveform = []
buffer = []
recording = False
total_frames = 0
fs = 44100  # Sample rate
pressed_keys = set()

def block(val, low=0, hi=50):
    blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    val = max(min(val, hi), low)
    if hi - low == 0:
        norm = 0.0
    else:
        norm = (val - low) / (hi - low)
    index = int(norm * (len(blocks) - 1))
    return blocks[index]

def callback(data, frames, time, status):
    global buffer, recording, total_frames, waveform
    if recording:
        buffer.extend(data.copy())
    total_frames += frames
    rms = np.sqrt(np.mean(data**2)) # Root Mean Square to estimate loudness
    loudness = (20 * np.log10(rms)) + 70.0 # Decibels, with silence at about 0
    
    waveform.append(float(loudness))
    term = os.get_terminal_size().columns//2
    if len(waveform) > term:
        print_wave = resample(waveform, term)
    else:
        print_wave = waveform
    width = term - len(print_wave)//2
    avg_l = sum(waveform)/len(waveform)
    min_l = (3*min(waveform)+avg_l)/4
    max_l = (3*max(waveform)+avg_l)/4
    
    string = ''.join(block(v, min_l, max_l) for v in print_wave)
    seconds = f' {float(total_frames)/fs:.2f} seconds'
    seconds = seconds + ' '*(width-len(seconds))
    console.print(seconds+string, end='\r')

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

def watch_record_audio(keys=None):
    global buffer, recording, total_frames, TEXT, waveform

    if keys == None:
        if platform.system() == 'Darwin': # Mac
            keys = (keyboard.Key.ctrl_l, keyboard.Key.cmd_l)
        else: # Windows and Linux
            keys = (keyboard.Key.ctrl_l, keyboard.Key.alt_l)

    controller = keyboard.Controller()
    
    with keyboard.Listener(on_press=on_press, on_release=on_release):
        while True:
            if not recording and all(k in pressed_keys for k in keys):
                console.print('')
                recording = True
                buffer = [] # Clear the buffer
                waveform = []

                stream = sd.InputStream(samplerate=fs, channels=2, callback=callback, blocksize=2205)
                stream.start()

                while all(k in pressed_keys for k in keys):
                    time.sleep(0.01)

                recording = False
                stream.stop()
                recorded_audio = np.array(buffer)
                filename = "temp.wav"
                write(filename, fs, recorded_audio)
                console.print('')
                if float(total_frames)/fs > 0.15: # Min time 0.15 sec
                    TEXT = transcribe_audio(filename)
                    console.print(TEXT, end='')
                total_frames = 0

                if SUBMIT_TEXT:
                    controller.press(keyboard.Key.enter)
                    controller.release(keyboard.Key.enter)
            
            time.sleep(0.1)
