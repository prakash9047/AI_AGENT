from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from typing import List, Dict, Optional
import json
import time

# Import enhanced tools
from tools import (
    save_tool,
    save_json_tool,
    advanced_search_tool,
    web_scrape_tool,
    article_extract_tool,
    pdf_extract_tool,
    arxiv_search_tool,
    multi_source_research_tool,
    wiki_tool  # Deprioritized but still available
)

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Enhanced response model
class ResearchResponse(BaseModel):
    topic: str = Field(description="The main research topic")
    summary: str = Field(description="A comprehensive summary of the research findings")
    key_points: List[str] = Field(description="Key points or findings from the research")
    sources: List[Dict[str, str]] = Field(description="Sources used in the research with title and URL")
    tools_used: List[str] = Field(description="Tools used during the research process")

# Setup LLM with higher temperature for more creative synthesis
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", 
    google_api_key=api_key,
    temperature=0.3,  # Slightly higher for better synthesis
    max_output_tokens=4096
)

# Enhanced system prompt prioritizing diverse sources
system_prompt = """
You are an advanced research assistant that produces comprehensive, accurate, and well-sourced information similar to ChatGPT, DeepSeek, and Perplexity's research mode.

RESEARCH GUIDELINES:
1. PRIORITIZE DIVERSE WEB SOURCES over Wikipedia. Use Wikipedia only as a supplementary source.
2. Gather information from multiple websites, academic sources, and specialized resources.
3. For each topic, conduct thorough research using multiple tools to ensure comprehensive coverage.
4. Synthesize information from different sources to create a coherent and complete understanding.
5. Include SPECIFIC URLs and titles for ALL sources used in your research.
6. Be objective and present multiple viewpoints when relevant.

RESEARCH PROCESS:
1. START WITH THE MULTI-SOURCE RESEARCH TOOL to gather broad information from diverse sources.
2. Use ADVANCED SEARCH to find specific information not covered by the initial research.
3. For specific websites, use WEB SCRAPING to extract more detailed information.
4. For academic topics, use ARXIV SEARCH to find scholarly articles.
5. Extract content from PDFs when necessary using the PDF EXTRACTION tool.
6. Use ARTICLE EXTRACTION for in-depth analysis of specific web pages.
7. Only use WIKIPEDIA as a supplementary source, not as your primary source.

Your output should be in valid JSON format with the following structure:
{{
    "topic": "The main research topic",
    "summary": "A comprehensive summary of research findings",
    "key_points": ["Key point 1", "Key point 2", "Key point 3", ...],
    "sources": [
        {{"title": "Source Title 1", "url": "https://source-url-1.com"}},
        {{"title": "Source Title 2", "url": "https://source-url-2.com"}},
        ...
    ],
    "tools_used": ["tool_name_1", "tool_name_2", ...]
}}

Remember, your goal is to provide high-quality, comprehensive research with diverse sources similar to ChatGPT and Perplexity.
"""

# Create prompt template with correct variables
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Include all tools with multi-source research prioritized
tools = [
    multi_source_research_tool,  # Prioritize this tool for comprehensive research
    advanced_search_tool,        # Secondary priority for search
    article_extract_tool,        # For detailed article extraction
    web_scrape_tool,             # For general web scraping
    arxiv_search_tool,           # For academic research
    pdf_extract_tool,            # For PDF extraction
    wiki_tool,                   # Deprioritized but still available
    save_tool,
    save_json_tool
]

# Create agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Create agent executor with higher timeout and iterations
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=15,  # Increased for more thorough research
    handle_parsing_errors=True,
    early_stopping_method="generate",  # Better handling of completion
    max_execution_time=300  # 5 minutes max execution time
)

def run_research():
    query = input("What can I help you research? ")
    
    print("\nResearching your topic using multiple web sources. This may take a few minutes...\n")
    start_time = time.time()
    
    try:
        # Execute the agent
        raw_response = agent_executor.invoke({"query": query})
        
        # Calculate research time
        research_time = time.time() - start_time
        print(f"\nResearch completed in {research_time:.2f} seconds")
        
        # Extract the response
        if isinstance(raw_response, dict) and "output" in raw_response:
            if isinstance(raw_response["output"], str):
                output_text = raw_response["output"]
            elif isinstance(raw_response["output"], list) and len(raw_response["output"]) > 0:
                if isinstance(raw_response["output"][0], dict) and "text" in raw_response["output"][0]:
                    output_text = raw_response["output"][0]["text"]
                else:
                    output_text = str(raw_response["output"][0])
            else:
                output_text = str(raw_response["output"])
        else:
            output_text = str(raw_response)
        
        # Try to parse the response as JSON
        try:
            # Find JSON in the output text if it's embedded
            json_start = output_text.find('{')
            json_end = output_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = output_text[json_start:json_end]
                result = json.loads(json_text)
            else:
                # If no JSON found, try direct parsing
                result = json.loads(output_text)
                
            # Print formatted response
            print("\n" + "="*50)
            print(f"RESEARCH RESULTS: {result.get('topic', 'Research Topic')}")
            print("="*50)
            print(f"\nSUMMARY:\n{result.get('summary', 'No summary available')}")
            
            if 'key_points' in result and result['key_points']:
                print("\nKEY POINTS:")
                for i, point in enumerate(result['key_points'], 1):
                    print(f"{i}. {point}")
            
            if 'sources' in result and result['sources']:
                print("\nSOURCES:")
                for i, source in enumerate(result['sources'], 1):
                    if isinstance(source, dict):
                        title = source.get('title', 'Untitled')
                        url = source.get('url', 'No URL provided')
                        print(f"{i}. {title}: {url}")
                    else:
                        print(f"{i}. {source}")
            
            if 'tools_used' in result and result['tools_used']:
                print("\nTOOLS USED:")
                for tool in result['tools_used']:
                    print(f"- {tool}")
            
            # Save results
            save_to_json = input("\nWould you like to save these results to a JSON file? (y/n): ").lower()
            if save_to_json == 'y':
                filename = input("Enter filename (default: research_output.json): ") or "research_output.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to {filename}")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print("Raw response:")
            print(output_text[:500] + "..." if len(output_text) > 500 else output_text)
            
            # Save the raw response
            save_raw = input("Would you like to save the raw response to a file? (y/n): ").lower()
            if save_raw == 'y':
                filename = input("Enter filename (default: raw_response.txt): ") or "raw_response.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(output_text)
                print(f"Raw response saved to {filename}")
                
            return {"error": "Unable to parse JSON response", "raw_output": output_text[:1000]}
            
    except Exception as e:
        print(f"Error during research: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    run_research()