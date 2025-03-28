# Advanced AI Research Assistant

A powerful research tool that produces comprehensive, accurate, and well-sourced information similar to ChatGPT, DeepSeek, and Perplexity's research mode. This assistant uses multiple tools to gather, synthesize, and present information from diverse sources.

## Table of Contents
- [Features](#features)
- [Tools Used](#tools-used)
- [Installation](#installation)
- [Required Libraries](#required-libraries)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Example](#example)
- [License](#license)
- [Contributing](#contributing)

## Features

- **Multi-source research**: Gathers information from multiple websites and specialized resources
- **Academic search**: Finds scholarly articles using ArXiv integration
- **Content extraction**: Extracts information from websites, articles, and PDFs
- **Comprehensive synthesis**: Combines information from multiple sources
- **Well-structured output**: Presents research in a clear, organized format
- **Source tracking**: Includes specific URLs and titles for all sources used

## Tools Used

| Tool | Purpose |
|------|---------|
| **Multi-source research** | Primary tool for gathering information from diverse sources |
| **Advanced search** | For finding specific information |
| **Article extraction** | For in-depth analysis of web pages |
| **Web scraping** | For general content extraction |
| **ArXiv search** | For academic research papers |
| **PDF extraction** | For analyzing PDF documents |
| **Wikipedia** | As a supplementary source |

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/advanced-ai-research-assistant.git
   cd advanced-ai-research-assistant
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root directory with your API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Required Libraries

The project relies on several Python libraries:

```
langchain
langchain-google-genai
python-dotenv
pydantic
requests
beautifulsoup4
arxiv
PyPDF2
newspaper3k (optional but recommended for better article extraction)
duckduckgo-search
```

## Usage

Run the main script to start the research assistant:

```bash
python main.py
```

The program will prompt you to enter a research query. After processing, it will display:
- A comprehensive summary of the findings
- Key points from the research
- Sources used (with URLs)
- Tools used in the research process

You can optionally save the research results to a JSON file.

## Output Structure

The research results are structured as follows:

```json
{
    "topic": "The main research topic",
    "summary": "A comprehensive summary of research findings",
    "key_points": ["Key point 1", "Key point 2", "Key point 3", ...],
    "sources": [
        {"title": "Source Title 1", "url": "https://source-url-1.com"},
        {"title": "Source Title 2", "url": "https://source-url-2.com"},
        ...
    ],
    "tools_used": ["tool_name_1", "tool_name_2", ...]
}
```

## Example

**Input:**
```
What can I help you research? Latest developments in quantum computing
```

**Output:**
```
RESEARCH RESULTS: Latest Developments in Quantum Computing
==================================================

SUMMARY:
[Comprehensive summary of quantum computing developments]

KEY POINTS:
1. [Key point about quantum advantage]
2. [Key point about error correction]
3. [Key point about new quantum algorithms]

SOURCES:
1. Quantum Computing Report: https://quantumcomputingreport.com/
2. Nature: Quantum Information: https://www.nature.com/subjects/quantum-information
...

TOOLS USED:
- multi_source_research_tool
- arxiv_search_tool
- article_extract_tool
```

## License

[Your license type here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<p align="center">
Built with ❤️ using Gemini 1.5 Pro
</p>
