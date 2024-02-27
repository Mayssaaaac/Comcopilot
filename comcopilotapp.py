from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from moviepy.editor import VideoFileClip
import parselmouth
from parselmouth.praat import call, run_file
import numpy as np
import noisereduce as nr
import os
import tempfile
import analysis_utils
from typing import Dict, Any
from scipy.io import wavfile
import logging

app = FastAPI()

TOTAL_CRITERIA = 5
PRAAT_SCRIPT_PATH = "script_content.praat"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/video/analysis/")
async def analyze_video(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as video_file:
            contents = await file.read()
            video_file.write(contents)
            video_file.flush()

        video_file_path = video_file.name
        audio_file_path = convert_video_to_audio(video_file_path)
        
        if not audio_file_path:
            logger.error("Failed to convert video to audio.")
            response_data = {"error": "Failed to convert video to audio."}
        else:
            analysis_results = perform_analysis(audio_file_path)
            response_data = prepare_response_data(analysis_results)
        
        cleanup_files([video_file_path, audio_file_path])
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.exception("Exception occurred during video analysis")
        raise HTTPException(status_code=500, detail=str(e))

def convert_video_to_audio(video_file_path: str) -> str:
    try:
        video_clip = VideoFileClip(video_file_path)
        audio_file_path = tempfile.mktemp(suffix=".wav")
        video_clip.audio.write_audiofile(audio_file_path, codec='pcm_s16le')
        video_clip.close()  # Ensure resources are released
        return audio_file_path
    except Exception as e:
        logger.exception("Error converting video to audio")
        return ""

def perform_analysis(audio_file_path: str) -> Dict[str, Any]:
    available_criteria_count = 5
    praat_script_path = PRAAT_SCRIPT_PATH
    path = os.path.dirname(audio_file_path)

    analysis_results = {
        "pitch": {"characteristic": "Error: Analysis failed"},
        "volume": {"characteristic": "Error: Analysis failed"},
        "silence": {"characteristic": "Error: Analysis failed"},
        "pronunciation": {"score": 0},
        "articulation": {"rate": 0},
        "overall_score": 0
    }

    try:
        sound = parselmouth.Sound(audio_file_path)
        
        pitch_analysis_result = analysis_utils.analyze_pitch(audio_file_path)
        pitch_result = "Error: Pitch analysis failed"
        if pitch_analysis_result and "pitch" in pitch_analysis_result:
            pitch_std_dev = pitch_analysis_result["pitch"]
            pitch_result = analysis_utils.classify_speaker(pitch_std_dev)

        volume_result = analysis_utils.analyze_volume(sound)
        silences = analysis_utils.analyze_silences(sound)
        silence_result = analysis_utils.classify_silences(silences)
        segments = analysis_utils.segment_audio(sound)
        pronunciation_score = analysis_utils.average_score(audio_file_path, praat_script_path, os.path.dirname(audio_file_path), 14)
        articulation_rate = analysis_utils.average_score(audio_file_path, praat_script_path, os.path.dirname(audio_file_path), 3)
        available_criteria_count = 5
        positive_criteria_count = (1 if pitch_result == "Balanced" else 0) + \
                                  (1 if volume_result == "Volume is ideal" else 0) + \
                                  (1 if silence_result == "normal" else 0) + \
                                  (pronunciation_score * 1) + \
                                  (1 if articulation_rate >= 3 else 0)
        print(f"Positive Criteria Count: {positive_criteria_count}, Available Criteria Count: {available_criteria_count}")
        overall_score = analysis_utils.calculate_score(positive_criteria_count, available_criteria_count)
        print(f"Calculated Overall Score: {overall_score}")
        analysis_results.update({
            "pitch": {"characteristic": pitch_result},
            "volume": {"characteristic": volume_result},
            "silence": {"characteristic": silence_result},
            "pronunciation": {"score": pronunciation_score},
            "articulation": {"rate": articulation_rate},
            "overall_score": {"score": overall_score}
        })
    except Exception as e:
        logger.exception("Error during analysis")

    return analysis_results


def prepare_response_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "criteria": {
            "pitch": analysis_results["pitch"],
            "volume": analysis_results["volume"],
            "silence": analysis_results["silence"],
            "pronunciation": analysis_results["pronunciation"],
            "articulation": analysis_results["articulation"],
            
        },
        "overall_score": analysis_results["overall_score"]
    }

def cleanup_files(file_paths: [str]):
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.exception(f"Error cleaning up file {path}")

