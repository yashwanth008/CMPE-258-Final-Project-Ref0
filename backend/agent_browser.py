from browser_use import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Setup Gemini for the Browser Agent
# Note: Browser Use works best with 'gemini-1.5-pro' for reasoning, 
# but 'flash' is faster. We use Flash for speed.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", # Use 2.0 or 1.5-flash
    api_key=os.getenv("GEMINI_API_KEY")
)

async def verify_rule_with_browser(rule_citation, sport="NFL"):
    print(f"🕵️ BROWSER AGENT: Verifying '{rule_citation}'...")
    
    task = f"""
    Go to google.com.
    Search for "{sport} Official Rulebook {rule_citation}".
    Click on the first official link (nfl.com, nba.com etc).
    Find the text of the rule.
    Return the exact text description of the foul.
    """
    
    agent = Agent(
        task=task,
        llm=llm,
        headless=False,  # ✅ TRUE = Invisible, FALSE = User sees the browser open!
    )
    
    try:
        result = await agent.run()
        return result.final_result
    except Exception as e:
        return f"Could not verify rule: {str(e)}"