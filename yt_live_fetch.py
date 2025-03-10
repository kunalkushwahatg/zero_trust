import subprocess
from io import BytesIO
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import time

class LiveExtraction:
    def __init__(self):
        pass

    def get_live_stream_url(self,youtube_url):
        command = ["yt-dlp", "-g", youtube_url]
        process = subprocess.run(command, capture_output=True, text=True)
        stream_url = process.stdout.strip()
        return stream_url
    def extract_audio_clip_as_waveform(self, youtube_url, duration,start_time=0):
        """
        Extracts a specific audio clip from a YouTube video as a waveform using streaming.

        Args:
            youtube_url (str): The URL of the YouTube video.
            start_time (int): Start time of the clip in seconds.
            duration (int): Duration of the clip in seconds.

        Returns:
            np.ndarray: The waveform array of the audio clip.
        """
        command = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", youtube_url,
            "-t", str(duration),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # Audio codec (WAV format)
            "-ar", "16000",  # Sample rate
            "-ac", "1",  # Mono
            "-f", "wav",  # WAV format
            "pipe:1"  # Output to pipe
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_data, _ = process.communicate()

        # Convert audio data to waveform
        audio = AudioSegment.from_wav(BytesIO(audio_data))
        waveform = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2 ** 15)
        return waveform

    def extract_video_audio_as_waveform(self, youtube_url):
        """
        Extracts the entire audio from a non-live YouTube video and returns the waveform array.

        Args:
            youtube_url (str): The URL of the YouTube video.

        Returns:
            np.ndarray: The waveform array of the audio.
        """
        temp_audio_file = "temp_audio.wav"
        command = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "wav",  # Convert to WAV format
            "-o", temp_audio_file,  # Output file
            youtube_url
        ]
        subprocess.run(command, check=True)
        audio = AudioSegment.from_file(temp_audio_file, format="wav")
        waveform = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2 ** 15)
        subprocess.run(["rm", temp_audio_file], check=True)

        return waveform


if __name__ == "__main__":
    le = LiveExtraction()
    stream_url = le.get_live_stream_url("https://www.youtube.com/watch?v=gadjsB5BkK4")
    while True:
        start_time = time.time()
        waveform = le.extract_audio_clip_as_waveform(stream_url, duration=10)
        print("wavefrom extarction time ",time.time()-start_time)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.6f} seconds")
        