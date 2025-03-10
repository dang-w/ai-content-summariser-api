import httpx
from bs4 import BeautifulSoup
import re

class URLExtractorService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def extract_content(self, url):
        """
        Extract the main content from a URL.

        Args:
            url (str): The URL to extract content from

        Returns:
            str: The extracted text content
        """
        try:
            response = await self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()

            # Get text and clean it
            text = soup.get_text()

            # Break into lines and remove leading/trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Remove blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text
        except Exception as e:
            raise Exception(f"Failed to extract content from URL: {str(e)}")
        finally:
            await self.client.aclose()