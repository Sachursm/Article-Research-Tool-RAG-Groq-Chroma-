from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from playwright.async_api import async_playwright
import asyncio
import pypdf
import re
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

# This module is used to load js heavy websites
async def scrape_with_playwright(url: str) -> Document:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",              # required in Docker
                "--disable-dev-shm-usage",   # prevents crashes on low memory
                "--disable-gpu",             # no GPU in HF CPU container
                "--single-process"           # lighter on 2 vCPU
            ]
        )
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
        except Exception:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)

        content = await page.inner_text("body")
        await browser.close()
        return Document(page_content=content, metadata={"source": url})

# To find if the url is a youtube video
def find_youtube_url(url: str) -> str:
    if re.search(r"(youtube\.com|youtu\.be)", url):
        return url

# To extract youtube video id from the url
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

# To get the title of youtube video
def get_video_title(url: str) -> str:
    try:
        yt = YouTube(url)
        return yt.title
    except Exception:
        return url 

# To get the transcript of youtube video
def get_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        return " ".join([t.text for t in transcript])
    except Exception as e:
        raise ValueError(f"Could not get transcript: {e}")


# Main function to scrape urls 
def scrape_urls(urls: list) -> list:
    result = []
    for url in urls:
        if find_youtube_url(url):
            title = get_video_title(url)
            video_id = extract_video_id(url)
            transcript = get_transcript(video_id)
            result.append(Document(
                page_content=transcript,
                metadata={"source": url, "title": title}
            ))
        else:
            loader = WebBaseLoader(
                web_paths=[url],
                header_template={"User-Agent": "Mozilla/5.0"}
            )
            data = loader.load()
            if len(data[0].page_content) < 500:
                doc = asyncio.run(scrape_with_playwright(url))
                result.append(doc)
            else:
                doc = Document(
                    page_content=data[0].page_content,
                    metadata=data[0].metadata
                )
                result.append(doc)
    return result

# Functions to extract text from PDF 
def extract_pdf(files: list)-> list:
    result = []
    for file in files:
        reader = pypdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        result.append(Document(page_content=text, 
                              metadata={"source": file.name}))
    return result


# Functions to extract text from txt files
def extract_txt(files: list) -> list:
    result = []
    for file in files:
        text = file.read().decode("utf-8")
        result.append(Document(page_content=text, 
                              metadata={"source": file.name}))
    return result