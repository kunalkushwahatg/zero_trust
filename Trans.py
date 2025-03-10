from youtube_transcript_api import YouTubeTranscriptApi
from SearchVerification import FactChecker
from api import serpapi, api

class YouTubeTranscriptProcessor:
    def __init__(self, url, text_size=20):
        """
        Initialize with YouTube URL and the desired number of words per line.
        """
        self.video_id = self.extract_video_id(url)
        self.text_size = text_size
        self.transcript = []
        self.processed_transcript = []

    @staticmethod
    def extract_video_id(url):
        """
        Extract the video ID from the YouTube URL.
        """
        if "v=" in url:
            return url.split("v=")[1]
        raise ValueError("Invalid YouTube URL format.")

    def fetch_transcript(self):
        """
        Fetch the transcript for the YouTube video.
        """
        try:
            self.transcript = YouTubeTranscriptApi.get_transcript(self.video_id)
        except Exception as e:
            raise RuntimeError(f"Error fetching transcript: {e}")

    def process_transcript(self):
        """
        Process the transcript into manageable lines with specified text size.
        """
        if not self.transcript:
            raise ValueError("Transcript is empty. Fetch it before processing.")

        current_line = []
        current_duration = 0
        current_start_time = self.transcript[0]['start']

        for entry in self.transcript:
            words = entry['text'].split()
            for word in words:
                current_line.append(word)
                current_duration += entry['duration'] / len(words)
                if len(current_line) >= self.text_size:
                    self.processed_transcript.append({
                        'text': ' '.join(current_line),
                        'start': current_start_time,
                        'duration': current_duration
                    })
                    current_line = []
                    current_duration = 0
                    current_start_time = entry['start'] + entry['duration']

        # Add the last line if there are remaining words
        if current_line:
            self.processed_transcript.append({
                'text': ' '.join(current_line),
                'start': current_start_time,
                'duration': current_duration
            })

    def get_processed_transcript(self):
        """
        Return the processed transcript.
        """
        return self.processed_transcript


# Example Usage
if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=-2k1rcRzsLA"
    processor = YouTubeTranscriptProcessor(url, text_size=20)

    # Fetch and process transcript
    processor.fetch_transcript()
    processor.process_transcript()

    # Print the processed transcript
    processed = processor.get_processed_transcript()
    fact = FactChecker(serpapi, api)
    counter = 0
    for entry in processed:

        print(entry['text'])
        query = entry['text']
        response = fact.fact_check_with_openai(query)
        #print(response)
        sentiment, claim_verification = fact.extract_json(response)
        print("Sentiment:", sentiment)
        print("Claim Verification:", claim_verification)

        print(f"Start Time: {entry['start']}")
        print(f"Duration: {entry['duration']}\n")
        
        if counter == 5:
            break
        counter += 1
    print(f"Total Lines: {len(processed)}")
