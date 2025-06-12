import re
from transformers import pipeline
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import pandas as pd
import gc
from bs4 import BeautifulSoup
import torch


def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


class YoutubeAnalysis:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sentiment_classifier = pipeline(
            "text-classification",
            model="Remicm/sentiment-analysis-model-for-socialmedia",
            device=device
        )
        self.spam_classifier = pipeline(
            "text-classification",
            model="MathewManoj/tinybert-spam-detector",
            device=device
        )
        self.sarcasm_classifier = pipeline(
            "text-classification",
            model="helinivan/english-sarcasm-detector",
            device=device
        )
        self.emotion_classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/bert-base-uncased-emotion",
            device=device
        )
        print("Models loaded successfully! (GPU used: {})".format(torch.cuda.is_available()))

    def select_comments(self, df_feedback, top_k=20):
        if len(df_feedback) > 200:
            df_feedback = df_feedback[df_feedback['sarcasm_label'] == 'LABEL_0']
        if len(df_feedback) > 100:
            df_feedback = df_feedback[df_feedback['spam_label'] == 'LABEL_0']
        if len(df_feedback) > 100:
            df_feedback = df_feedback[df_feedback['word_count'] > 5]
        if len(df_feedback) > 100:
            df_feedback = df_feedback[df_feedback['word_count'] < 50]
        if len(df_feedback) > 100:
            df_feedback = df_feedback[df_feedback['emotion'].isin(['joy', 'surprise'])]
        if len(df_feedback) > 100:
            df_feedback = df_feedback.sort_values(by='votes', ascending=False)
        if len(df_feedback) > top_k:
            df_feedback = df_feedback.head(top_k)


        return df_feedback['text'].tolist()

    def download_comments(self, url):
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
        comment_data = []
        for comment in comments:
            cleaned_comment = {
                'text': comment.get('text', ''),
                'votes': comment.get('votes', ''),
                'reply_count': comment.get('reply_count', ''),
                'heart': comment.get('heart', '')
            }
            comment_data.append(cleaned_comment)

        df_comments = pd.DataFrame(comment_data)
        return df_comments.drop_duplicates(subset='text')
    
    def clean_and_filter_comments(self, df):
        df['text'] = df['text'].apply(clean_text)
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
        df = df[(df['word_count'] > 2) & (df['word_count'] < 500)]


        # ✅ Use preloaded models
        results = [self.sentiment_classifier(text)[0] for text in df['text']]
        df['sentiment_probability'] = [res['score'] for res in results]
        df['sentiment'] = [res['label'] for res in results]

        spam_results = [self.spam_classifier(text)[0] for text in df['text']]
        df['spam_probability'] = [res['score'] for res in spam_results]
        df['spam_label'] = [res['label'] for res in spam_results]

        sarcasm_results = [self.sarcasm_classifier(text)[0] for text in df['text']]
        df['sarcasm_probability'] = [res['score'] for res in sarcasm_results]
        df['sarcasm_label'] = [res['label'] for res in sarcasm_results]

        emotion_results = [self.emotion_classifier(text)[0] for text in df['text']]
        df['emotion_probability'] = [res['score'] for res in emotion_results]
        df['emotion'] = [res['label'] for res in emotion_results]

        return df  # <-- Add this line

    def stats(self,df):
        n_positive = df[df['sentiment'] == 'LABEL_1'].shape[0]
        n_negative = df[df['sentiment'] == 'LABEL_0'].shape[0]

        n_sarcasm = df[df['sarcasm_label'] == 'LABEL_1'].shape[0]
        n_not_sarcasm = df[df['sarcasm_label'] == 'LABEL_0'].shape[0]

        n_spam = df[df['spam_label'] == 'LABEL_1'].shape[0]
        n_not_spam = df[df['spam_label'] == 'LABEL_0'].shape[0]

        n_sadness = df[df['emotion'] == 'sadness'].shape[0]
        n_joy = df[df['emotion'] == 'joy'].shape[0]
        n_love = df[df['emotion'] == 'love'].shape[0]
        n_anger = df[df['emotion'] == 'anger'].shape[0]
        n_fear = df[df['emotion'] == 'fear'].shape[0]
        n_surprise = df[df['emotion'] == 'surprise'].shape[0]

    #get the percentage of each emotion out of 100
        emotion_stats = {
            "sadness": (n_sadness / df.shape[0]) * 100,
            "joy": (n_joy / df.shape[0]) * 100,
            "love": (n_love / df.shape[0]) * 100,
            "anger": (n_anger / df.shape[0]) * 100,
            "fear": (n_fear / df.shape[0]) * 100,
            "surprise": (n_surprise / df.shape[0]) * 100
        }


    #get the percentage of spam and not spam out of 100
        spam_stats = {
            "spam": (n_spam / df.shape[0]) * 100,
            "not_spam": (n_not_spam / df.shape[0]) * 100
        }

        sentiment_stats = {
            "positive": (n_positive / df.shape[0]) * 100,
            "negative": (n_negative / df.shape[0]) * 100
        }

        sarcasm_stats = {
            "sarcasm": (n_sarcasm / df.shape[0]) * 100,
            "not_sarcasm": (n_not_sarcasm / df.shape[0]) * 100
        }

    # return the stats

        return { "spam_stats": spam_stats, "sentiment_stats": sentiment_stats, "sarcasm_stats": sarcasm_stats , "emotion_stats": emotion_stats}
    
    def analyze_comments(self, url):
        df = self.download_comments(url)
        if df.empty:
            return "No comments found or all comments were filtered out."

        df = self.clean_and_filter_comments(df)  # <-- Assign the filtered df

        if df.empty:
            return "No comments left after cleaning and filtering."

        stats = self.stats(df)
        return stats
    
if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=example"  # Replace with your YouTube video URL
    analysis = YoutubeAnalysis()
    result = analysis.analyze_comments(url)
    print(result)
    gc.collect()  # Clean up memory
