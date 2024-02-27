from moviepy.editor import VideoFileClip
import parselmouth
from parselmouth.praat import call, run_file
import numpy as np
import noisereduce as nr
import os
import tempfile
from scipy.io import wavfile

TOTAL_CRITERIA = 5

def analyze_pitch(audio_data):
    try:
        sr, sound_values = wavfile.read(audio_data)
        if sound_values.ndim > 1:  
            sound_values = sound_values[:,0]

        sound = parselmouth.Sound(sound_values, sampling_frequency=sr)
        
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        
        pitch_values_filtered = pitch_values[(pitch_values > 0) & (pitch_values >= 75) & (pitch_values <= 400)]
        if len(pitch_values_filtered) == 0:
            print("No valid pitch values found.")
            return None  

        std_dev = np.std(pitch_values_filtered)
        return {"pitch": std_dev}
    except Exception as e:
        print("Exception analyzepitch")
        print(e)
        return None



def classify_speaker(std_dev):
    if std_dev is None:
        return "Unknown"  
    if 10 <= std_dev <= 60:
        return "Balanced"
    else:
        return "Unbalanced"

def analyze_volume(audio_data):
    snd = parselmouth.Sound(audio_data)

    intensity = snd.to_intensity()
    average_intensity = np.mean(intensity.values) if intensity.values.size > 0 else None
    if average_intensity is None:
        print("Exception volume")
        return "Unknown volume"  
    
    low_threshold = 45  
    high_threshold = 75 

    if average_intensity < low_threshold:
        return "Volume too low"
    elif average_intensity > high_threshold:
        return "Volume too loud"
    else:
        return "Volume is ideal"    

def analyze_silences(sound, noise_reduction=True, silence_threshold=40, min_silence_duration=0.5):
    try:
        if noise_reduction:
            audio_data = sound.values[0]  
            sampling_rate = sound.sampling_frequency
            noise_clip = audio_data[:int(sampling_rate * 0.5)]  
            reduced_noise_audio = nr.reduce_noise(y=audio_data, y_noise=noise_clip, sr=sampling_rate)
            sound = parselmouth.Sound(reduced_noise_audio, sampling_rate)

        intensity = sound.to_intensity()
        intensity_values = intensity.values[0]
        times = intensity.xs()

        silences = []
        current_silence = None
        for time, value in zip(times, intensity_values):
            if value < silence_threshold:
                if current_silence is None:
                    current_silence = [time, None]
                continue
            if current_silence is not None:
                current_silence[1] = time if current_silence[1] is None else current_silence[1]
                if current_silence[1] - current_silence[0] >= min_silence_duration:
                    silences.append(current_silence)
                current_silence = None

        if current_silence is not None and current_silence[1] is None:
            current_silence[1] = times[-1]  
            if current_silence[1] - current_silence[0] >= min_silence_duration:
                silences.append(current_silence)

        return silences
    except Exception as e:
        print(f"Error in analyze_silences: {e}")
        return []


def classify_silences(silences, total_duration_threshold=15, initial_delay_threshold=15, long_silence_threshold=10):
    if not silences:  
        return "normal" 

    total_silence_duration = sum(end - start for start, end in silences)
    longest_silence = max((end - start for start, end in silences), default=0)
    initial_delay = silences[0][0] if silences and silences[0][0] == 0 else 0

    if longest_silence >= long_silence_threshold:
        return "silence too long"
    if total_silence_duration > total_duration_threshold:
        return "too much silence"
    if initial_delay >= initial_delay_threshold:
        return "delay"
    else:
        return "normal"

def segment_audio(sound, segment_length=15.0):
    try:
        segments = []
        for start_time in np.arange(0, sound.duration, segment_length):
            end_time = start_time + segment_length
            if end_time > sound.duration:
                end_time = sound.duration
            segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
            segments.append(segment)
        return segments
    except Exception as e:
        print(f"Error in segment_audio: {e}")
        return []


def analyze_segment(segment, praat_script_path, path):
    praat_script_path = "script_content.praat"
    try:
        with open(praat_script_path, "r") as file:
            praat_script_content = file.read()
    except FileNotFoundError:
        return None 
    with tempfile.NamedTemporaryFile("w", delete=False) as script_file:
        script_file.write(praat_script_content)
        temp_script_path = script_file.name
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file_path = tmp.name
        segment.save(audio_file_path, "WAV")
        objects = run_file(temp_script_path, -20, 2, 0.3, "yes", audio_file_path, path, 80, 400, 0.01, capture_output=True)
        os.remove(audio_file_path)
    
    os.remove(temp_script_path) 
    z1 = str(objects[1])
    z2 = z1.strip().split()
    return z2

def average_score(audio_file_path: str, praat_script_path: str, path: str, score_index: int):
    praat_script_path = "script_content.praat"    
    try:
        with open(praat_script_path, "r") as file:
            praat_script_content = file.read()
    except FileNotFoundError:
        print("scriptissue")
        return None  
    with tempfile.NamedTemporaryFile("w", delete=False) as script_file:
        script_file.write(praat_script_content)
        sourcerun = script_file.name  
    sound = parselmouth.Sound(audio_file_path)
    segments = segment_audio(sound)
    scores = []
    for segment in segments:
        z2 = analyze_segment(segment, sourcerun, path)
        if len(z2) > max(14, score_index):
            scores.append(float(z2[score_index]))
    if not scores:  
        return None  
    average_score = np.mean(scores)
    os.remove(sourcerun)
    return average_score


def calculate_score(positive_criteria_count, available_criteria_count):
    if available_criteria_count == 0:
        return 0
    else:
        return (positive_criteria_count / available_criteria_count) * 100
