try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    import requests

from bs4 import BeautifulSoup
import re
import logging

logger = logging.getLogger(__name__)

class URLExtractorService:
    async def extract_content(self, url: str) -> str:
        """Extract the main content from a URL."""
        try:
            if AIOHTTP_AVAILABLE:
                return await self._extract_with_aiohttp(url)
            else:
                return self._extract_with_requests(url)
        except Exception as e:
            logger.error(f"Error extracting content from URL {url}: {str(e)}")
            return ""

    async def _extract_with_aiohttp(self, url: str) -> str:
        """Extract content using aiohttp."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return ""

                html = await response.text()
                return self._parse_html(html)

    def _extract_with_requests(self, url: str) -> str:
        """Extract content using requests as fallback."""
        response = requests.get(url)
        if response.status_code != 200:
            return ""

        html = response.text
        return self._parse_html(html)

    def _parse_html(self, html: str) -> str:
        """Parse HTML and extract main content."""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove elements that typically contain comments or irrelevant content
        for element in soup.select('footer, .comments, #comments, .comment, .respond, .reply, .sidebar, nav, header, script, style, [id*=comment], [class*=comment]'):
            element.decompose()

        # Try to find the main content using common article containers
        main_content = None

        # Look for article tag first
        if soup.find('article'):
            main_content = soup.find('article')
        # Then try common content div classes/ids
        elif soup.find(class_=re.compile(r'(content|post|article|entry)(-body|-content|-text)?$', re.I)):
            main_content = soup.find(class_=re.compile(r'(content|post|article|entry)(-body|-content|-text)?$', re.I))
        # Then try main tag
        elif soup.find('main'):
            main_content = soup.find('main')

        if main_content:
            # Extract text from the main content
            text = main_content.get_text(separator=' ', strip=True)
        else:
            # Fallback to body if no main content container is found
            text = soup.body.get_text(separator=' ', strip=True)

        # Clean up the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'(\.|\?|!)\s+', r'\1\n\n', text)  # Add paragraph breaks after sentences

        # Remove common web page boilerplate text
        text = re.sub(r'Skip to (content|main).*?»', '', text)
        text = re.sub(r'Search for:.*?Search', '', text)
        text = re.sub(r'Menu.*?Resources', '', text, flags=re.DOTALL)

        # Remove comment sections (often start with phrases like "X responses to")
        text = re.sub(r'\d+ responses to.*?$', '', text, flags=re.DOTALL)

        # Remove form fields and subscription prompts
        text = re.sub(r'(Your email address will not be published|Required fields are marked).*?$', '', text, flags=re.DOTALL)

        return text