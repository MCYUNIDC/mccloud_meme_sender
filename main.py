class MemeSender:
    def __init__(self, meme_api_url):
        self.meme_api_url = meme_api_url

    def get_random_meme(self):
        # Logic to fetch a random meme from the API
        response = requests.get(self.meme_api_url)
        if response.status_code == 200:
            return response.json().get('meme_url')
        return None

    def send_meme(self, recipient, meme_url):
        # Logic to send the meme to the recipient
        print(f"Sending meme to {recipient}: {meme_url}")

    def run(self, recipient):
        meme_url = self.get_random_meme()
        if meme_url:
            self.send_meme(recipient, meme_url)