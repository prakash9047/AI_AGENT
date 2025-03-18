from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import json
import re
import os
import time
import random
from urllib.parse import urlparse

# Original tools
def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
        
    return f"Data successfully saved to {filename}"

def save_to_json(data: dict, filename: str = "research_output.json"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["timestamp"] = timestamp
    
    # Create file if it doesn't exist
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([], f)
    
    # Read existing data
    with open(filename, "r", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
    
    # Append new data
    existing_data.append(data)
    
    # Write back to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2)
    
    return f"Data successfully saved to {filename}"

# Enhanced Web Search Tools
def search_web(query: str, num_results: int = 5) -> str:
    """Search multiple search engines and aggregate results"""
    try:
        # Import here to avoid dependency issues
        from duckduckgo_search import DDGS
        
        results = []
        
        # DuckDuckGo search
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(query, max_results=num_results))
            for result in ddg_results:
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("body", ""),
                    "url": result.get("href", ""),
                    "source": "DuckDuckGo"
                })
        
        # Format the results
        formatted_results = ""
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. {result['title']}\n"
            formatted_results += f"   URL: {result['url']}\n"
            formatted_results += f"   Snippet: {result['snippet']}\n\n"
        
        return formatted_results
    except ImportError:
        # Fallback to simpler implementation
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Error during web search: {str(e)}"

# Improved web scraping tool
def scrape_webpage(url: str) -> str:
    """Scrape content from a webpage with improved error handling and rate limiting"""
    try:
        # Rate limiting to be respectful to websites
        time.sleep(1 + random.random())
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/",
            "DNT": "1"
        }
        
        # Parse the domain from the URL
        domain = urlparse(url).netloc
        
        # Add domain-specific headers/cookies for certain sites
        if "medium.com" in domain:
            headers["Cookie"] = "medium-tracking-id=123; uid=abc123"
        elif "github.com" in domain:
            headers["Accept"] = "application/json"
            
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Add source information
        text = f"Source URL: {url}\n\n" + text
        
        # Limit text length to prevent overwhelming the model
        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length] + "...[truncated]"
        
        return text
    except Exception as e:
        return f"Error scraping webpage {url}: {str(e)}"

# Smart article extraction
def extract_article_content(url: str) -> str:
    """Extract main article content from a webpage using smarter extraction techniques"""
    try:
        # Rate limiting
        time.sleep(1 + random.random())
        
        # Try to use more advanced extraction if available
        try:
            from newspaper import Article
            
            article = Article(url)
            article.download()
            article.parse()
            
            # Get metadata
            result = f"Title: {article.title}\n"
            result += f"Authors: {', '.join(article.authors)}\n" if article.authors else ""
            result += f"Publication Date: {article.publish_date}\n" if article.publish_date else ""
            result += f"Source URL: {url}\n\n"
            
            # Get content
            result += article.text
            
            return result
        except ImportError:
            # Fallback to simpler extraction
            return scrape_webpage(url)
    except Exception as e:
        return f"Error extracting article content from {url}: {str(e)}"

# Multiple source research function
def research_topic(query: str) -> str:
    """Research a topic by gathering information from multiple sources"""
    try:
        results = []
        
        # Step 1: Get initial search results
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        search_results = search.run(query)
        
        # Extract URLs from search results
        urls = re.findall(r'https?://[^\s]+', search_results)
        urls = [url.rstrip('.,;:)') for url in urls]
        
        # Only use first few valid URLs
        valid_urls = []
        for url in urls:
            # Filter out certain domains
            domain = urlparse(url).netloc
            if (
                domain and
                "wikipedia.org" not in domain and  # Deprioritize Wikipedia
                "youtube.com" not in domain and    # Skip video sites
                "facebook.com" not in domain and   # Skip social media
                "twitter.com" not in domain and
                ".gov" not in domain               # Skip government sites for research
            ):
                valid_urls.append(url)
                if len(valid_urls) >= 3:
                    break
        
        # Research message
        research_message = f"Researching: {query}\n\n"
        research_message += "Sources consulted:\n"
        
        # Scrape the valid URLs
        source_texts = []
        for i, url in enumerate(valid_urls, 1):
            try:
                # Use smarter extraction
                article_content = extract_article_content(url)
                source_texts.append(f"Source {i}: {url}\n\n{article_content}")
                research_message += f"{i}. {url}\n"
            except Exception as e:
                research_message += f"{i}. {url} - Error: {str(e)}\n"
        
        # Add search results too
        source_texts.append(f"Search Results:\n{search_results}")
        
        combined_research = research_message + "\n\n" + "\n\n---\n\n".join(source_texts)
        
        # Limit total length
        max_length = 20000
        if len(combined_research) > max_length:
            combined_research = combined_research[:max_length] + "...[truncated]"
        
        return combined_research
    except Exception as e:
        return f"Error during topic research: {str(e)}"

# Academic Search Tools
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search for academic papers on arXiv"""
    try:
        import arxiv
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in search.results():
            results.append({
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary,
                "url": paper.pdf_url,
                "published": paper.published.strftime("%Y-%m-%d")
            })
        
        # Format results
        formatted_results = "ArXiv Research Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. {result['title']}\n"
            formatted_results += f"   Authors: {', '.join(result['authors'])}\n"
            formatted_results += f"   Published: {result['published']}\n"
            formatted_results += f"   URL: {result['url']}\n"
            formatted_results += f"   Summary: {result['summary']}\n\n"
        
        return formatted_results
    except ImportError:
        return "Error: arxiv library not installed. Run 'pip install arxiv' to enable academic search."
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"

# PDF extraction tool
def extract_text_from_pdf_url(url: str) -> str:
    """Extract text from a PDF at the given URL"""
    try:
        import io
        import PyPDF2
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        max_pages = min(20, len(pdf_reader.pages))
        
        for page_num in range(max_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
        
        # Add source
        text = f"Source PDF: {url}\n\n" + text
        
        # Limit text length
        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length] + "...[truncated]"
        
        return text
    except ImportError:
        return "Error: PyPDF2 library not installed. Run 'pip install PyPDF2' to enable PDF extraction."
    except Exception as e:
        return f"Error extracting PDF content: {str(e)}"

# Define all tools
save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

save_json_tool = Tool(
    name="save_json_to_file",
    func=save_to_json,
    description="Saves structured research data to a JSON file.",
)

advanced_search_tool = Tool(
    name="search_web",
    func=search_web,
    description="Search the web for up-to-date information on any topic.",
)

web_scrape_tool = Tool(
    name="scrape_webpage",
    func=scrape_webpage,
    description="Scrape and extract content from a specific webpage URL.",
)

article_extract_tool = Tool(
    name="extract_article",
    func=extract_article_content,
    description="Extract main article content, title, and metadata from a webpage URL.",
)

pdf_extract_tool = Tool(
    name="extract_pdf",
    func=extract_text_from_pdf_url,
    description="Extract text from a PDF at the given URL.",
)

arxiv_search_tool = Tool(
    name="search_academic_papers",
    func=search_arxiv,
    description="Search for academic papers on arXiv.",
)

multi_source_research_tool = Tool(
    name="research_topic",
    func=research_topic,
    description="Conduct comprehensive research on a topic by gathering information from multiple web sources."
)

# Deprioritized Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wiki_tool = Tool(
    name="wikipedia_search",
    func=WikipediaQueryRun(api_wrapper=api_wrapper).run,
    description="Search Wikipedia only if you need general background information.",
)