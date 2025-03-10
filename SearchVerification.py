import requests
from openai import OpenAI
import json
import re

class FactChecker:
    def __init__(self, serpapi_key, openai_api_key):
        self.serpapi_key = serpapi_key
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)

    def fetch_current_data(self, query):
        """Fetch live search results using SerpAPI."""
        serpapi_url = "https://serpapi.com/search"
        params = {
            "q": query,
            "hl": "en",  # Language
            "gl": "in",  # Geolocation
            "api_key": self.serpapi_key,
        }
        response = requests.get(serpapi_url, params=params)
        if response.status_code == 200:
            results = response.json()
            if "organic_results" in results and results["organic_results"]:
                # Extract the top snippet or result
                top_result = results["organic_results"][0]
                return top_result.get("snippet", "No snippet available.")
        return "No results found."

    def fact_check_with_openai(self, query):
        """Perform fact-checking using OpenAI and live data."""
        current_info = self.fetch_current_data(query)
        if current_info == "No results found.":
            return "Unable to fetch current data for fact-checking."

        prompt = '''
        You are an advanced AI assistant specializing in text classification. Perform the following tasks on the given text:

        1. **Sentiment Analysis**: Analyze the sentiment of the text and classify it into one of the following categories:
           - Ultra Negative
           - Negative
           - Neutral
           - Positive
           - Ultra Positive

        2. **Fact Check**: Determine if the text contains a claim that can be evaluated as:
           - True
           - False
           - Neutral (if no claim is made or it is not verifiable based on available information).

        Use the SERP API search results for factual verification and make a decision based only on the available evidence.

        Query: {query}
        Current Info: {current_info}

        Respond in JSON format:
        {{
          "sentiment": "<one of: Ultra Negative, Negative, Neutral, Positive, Ultra Positive>",
          "claim_verification": "<one of: True, False, Neutral>"
        }}
        '''

        prompt = prompt.format(query=query, current_info=current_info)

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            store=True,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    @staticmethod
    def extract_json(response):
        """Extract sentiment and claim verification from the OpenAI response."""
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            response = match.group(0)
        else:
            return "No JSON found", "Error"

        try:
            response_dict = json.loads(response)
            sentiment = response_dict.get("sentiment", "Sentiment not available.")
            claim_verification = response_dict.get("claim_verification", "Claim verification not available.")
            return sentiment, claim_verification

        except json.JSONDecodeError:
            return "Error parsing JSON", "Error"

        except Exception as e:
            return f"Unexpected error: {str(e)}", "Error"


# Example Usage
if __name__ == "__main__":
    SERPAPI_KEY = ""
    OPENAI_API_KEY = ""
    
    fact_checker = FactChecker(SERPAPI_KEY, OPENAI_API_KEY)
    
    query = "I will make a machine that if input we feed in potato it will give us gold"
    response = fact_checker.fact_check_with_openai(query)
    print("Raw Response:", response)
    
    sentiment, claim_verification = FactChecker.extract_json(response)
    print("Sentiment:", sentiment)
    print("Claim Verification:", claim_verification)


