import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from groq import Groq
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Code Review & Rewrite Agent")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model configuration
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.3
MAX_TOKENS = 4096
TOP_P = 0.9


# Request models
class CodeReviewRequest(BaseModel):
    code: str
    language: str = "python"
    focus_areas: Optional[list[str]] = None
    mode: str = "developer"  # "developer" or "beginner"


class CodeRewriteRequest(BaseModel):
    code: str
    language: str = "python"
    instructions: Optional[str] = None


class CodeScoreRequest(BaseModel):
    code: str
    language: str = "python"


# ---------- Utility Functions ----------

def parse_review_response(review_text: str) -> dict:
    """Extract and categorize feedback from LLM-generated review text by priority levels."""
    categories = {
        "critical": {"pattern": r"(?:Critical Issues?|CRITICAL)(.*?)(?=(?:High Priority|HIGH|Medium Priority|MEDIUM|Low Priority|LOW|$))", "count": 0, "issues": []},
        "high": {"pattern": r"(?:High Priority|HIGH)(.*?)(?=(?:Medium Priority|MEDIUM|Low Priority|LOW|$))", "count": 0, "issues": []},
        "medium": {"pattern": r"(?:Medium Priority|MEDIUM)(.*?)(?=(?:Low Priority|LOW|$))", "count": 0, "issues": []},
        "low": {"pattern": r"(?:Low Priority|LOW)(.*?)$", "count": 0, "issues": []},
    }

    for key, cat in categories.items():
        match = re.search(cat["pattern"], review_text, re.DOTALL | re.IGNORECASE)
        if match:
            section = match.group(1).strip()
            issues = [line.strip("- ").strip() for line in section.split("\n") if line.strip() and line.strip() != "-"]
            cat["issues"] = issues
            cat["count"] = len(issues)

    return {
        "critical": {"count": categories["critical"]["count"], "issues": categories["critical"]["issues"]},
        "high": {"count": categories["high"]["count"], "issues": categories["high"]["issues"]},
        "medium": {"count": categories["medium"]["count"], "issues": categories["medium"]["issues"]},
        "low": {"count": categories["low"]["count"], "issues": categories["low"]["issues"]},
        "total_issues": sum(cat["count"] for cat in categories.values()),
    }


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def serve_login():
    """Serve the login page."""
    try:
        file_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "login.html")
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Login page not found")


@app.get("/tool", response_class=HTMLResponse)
async def serve_tool():
    """Serve the main tool page (index.html) after user login."""
    try:
        file_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Tool page not found")


@app.post("/review")
async def review_code(request: CodeReviewRequest):
    """Handle code review requests from the frontend using the Groq API."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code input cannot be empty")

    focus = ""
    if request.focus_areas:
        focus = f"\nFocus especially on these areas: {', '.join(request.focus_areas)}"

    prompt = f"""You are an expert code reviewer. Analyze the following {request.language} code and provide a thorough review.{focus}

Categorize your findings into these severity levels:
- **Critical Issues**: Bugs, security vulnerabilities, or errors that will cause failures
- **High Priority**: Performance problems, major code smells, or significant best-practice violations
- **Medium Priority**: Code style issues, minor optimizations, or readability improvements
- **Low Priority**: Suggestions, nice-to-haves, or minor formatting issues

For each issue found:
1. Describe the problem clearly
2. Explain why it matters
3. Suggest a specific fix with code if applicable

Code to review:
```{request.language}
{request.code}
```

Provide your review in a well-structured markdown format."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
        )

        review_text = chat_completion.choices[0].message.content
        parsed = parse_review_response(review_text)

        return {
            "review": review_text,
            "summary": parsed,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rewrite")
async def rewrite_code(request: CodeRewriteRequest):
    """Handle code rewrite requests — refactor and optimize the submitted code."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code input cannot be empty")

    extra_instructions = ""
    if request.instructions:
        extra_instructions = f"\nAdditional instructions: {request.instructions}"

    prompt = f"""You are an expert software engineer. Rewrite and optimize the following {request.language} code.{extra_instructions}

Your rewritten code should:
1. Fix all bugs and security issues
2. Optimize performance
3. Follow best practices and clean code principles
4. Include clear, concise comments
5. Be production-ready

Return ONLY the rewritten code inside a single code block with the language specified. After the code block, provide a brief summary of the changes made in a "## Changes Made" section.

Original code:
```{request.language}
{request.code}
```"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
        )

        rewrite_text = chat_completion.choices[0].message.content
        return {"rewritten": rewrite_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)