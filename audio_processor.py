
import webrtcvad

class AudioProcessor:
    def __init__(self):
        self.vad = webrtcvad.Vad(3)

        #sample rate is 16000 Hz as openai only supports 16kHz
        self.sample_rate = 16000

        self.frame_duration = 30  # one frame will consist of 30ms of audio
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000) * 2  # bytes in a frame
        
        
        self.chunk_duration = 1  # seconds of audio in a chunk (1 second)
        self.chunk_size = int(self.sample_rate * 2 * 2)  # 2 seconds of audio (16000Hz * 2s * 2bytes)

        #create a buffer to store audio data and a buffer to store speech data
        self.buffer = bytearray()
        self.speech_buffer = bytearray()
        self.speech_frames = 0
        self.total_frames = 0

    def add_audio(self, pcm_data):
        # Add the new audio data to the buffer
        self.buffer.extend(pcm_data)

    def get_speech_chunks(self):
        '''
        This function will return a list of chunks of audio data that contain speech.
        '''
        chunks = []
        while len(self.buffer) >= self.frame_size:
            frame = self.buffer[:self.frame_size]
            del self.buffer[:self.frame_size]
            self.total_frames += 1
            
            # Check if the frame contains speech or not
            if self.vad.is_speech(frame, self.sample_rate):
                self.speech_frames += 1
                self.speech_buffer.extend(frame)
                
            else:
                if len(self.speech_buffer) > 0:
                    self.speech_buffer.extend(frame)
                    

            # Check if we've accumulated 2 seconds of audio
            if len(self.speech_buffer) >= self.chunk_size:
                chunk = bytes(self.speech_buffer[:self.chunk_size])
                remaining = self.speech_buffer[self.chunk_size:]
                self.speech_buffer = bytearray(remaining)

                print(f"Speech frames : {self.speech_frames}"," out of ",self.total_frames)
                if self.speech_frames > 20:
                    chunks.append(chunk)
                self.speech_frames = 0
                self.total_frames = 0
                
        return chunks