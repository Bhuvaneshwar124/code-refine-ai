import os
import re
import json
import base64
import tempfile
from io import BytesIO
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from groq import Groq
from dotenv import load_dotenv
import uvicorn
import requests
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Code Review & Rewrite Agent")


# ============ VULNERABILITY KNOWLEDGE BASE (RAG) ============
VULNERABILITY_KB = [
    {
        "id": "sql_injection",
        "name": "SQL Injection",
        "severity": "Critical",
        "description": "SQL injection occurs when untrusted data is sent to an interpreter as part of a command or query.",
        "patterns": ["execute", "cursor.execute", "raw SQL", "f-string SQL", "% SQL", ".format(", "SELECT *", "WHERE", "INSERT INTO", "DELETE FROM", "UPDATE SET"],
        "examples": [
            "cursor.execute(f\"SELECT * FROM users WHERE id = {user_id}\")",
            "query = \"SELECT * FROM users WHERE name = '\" + name + \"'\""
        ],
        "fix": "Use parameterized queries or prepared statements. Example: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        "beginner_explanation": "Imagine a form asking for your name. If someone types special database commands instead of their name, they could access or delete all your data. Parameterized queries treat user input as data, not commands.",
        "cwe": "CWE-89",
        "owasp": "A03:2021 Injection"
    },
    {
        "id": "xss",
        "name": "Cross-Site Scripting (XSS)",
        "severity": "High",
        "description": "XSS attacks inject malicious scripts into web pages viewed by other users.",
        "patterns": ["innerHTML", "document.write", "eval(", "dangerouslySetInnerHTML", "v-html", "ng-bind-html", "<script>", "onclick=", "onerror="],
        "examples": [
            "element.innerHTML = userInput",
            "document.write('<div>' + userData + '</div>')"
        ],
        "fix": "Sanitize and encode user input before rendering. Use textContent instead of innerHTML. Use libraries like DOMPurify.",
        "beginner_explanation": "When you display user input on a webpage without cleaning it, attackers can inject JavaScript that steals cookies or redirects users. Always sanitize input before displaying it.",
        "cwe": "CWE-79",
        "owasp": "A03:2021 Injection"
    },
    {
        "id": "hardcoded_secrets",
        "name": "Hardcoded Secrets/Credentials",
        "severity": "Critical",
        "description": "Sensitive data like passwords, API keys, or tokens stored directly in source code.",
        "patterns": ["password =", "api_key =", "secret =", "token =", "AWS_ACCESS", "PRIVATE_KEY", "apiKey:", "Bearer ", "Basic "],
        "examples": [
            "API_KEY = 'sk-abc123xyz789'",
            "password = 'admin123'"
        ],
        "fix": "Use environment variables, secret managers (AWS Secrets Manager, HashiCorp Vault), or .env files excluded from version control.",
        "beginner_explanation": "Putting passwords in your code is like writing your PIN on your debit card. Anyone who sees your code gets access. Use environment variables instead.",
        "cwe": "CWE-798",
        "owasp": "A02:2021 Cryptographic Failures"
    },
    {
        "id": "unsafe_deserialization",
        "name": "Unsafe Deserialization",
        "severity": "Critical",
        "description": "Deserializing untrusted data can lead to remote code execution, replay attacks, or privilege escalation.",
        "patterns": ["pickle.load", "yaml.load", "marshal.load", "eval(", "exec(", "unserialize", "JSON.parse", "ObjectInputStream"],
        "examples": [
            "data = pickle.load(user_file)",
            "config = yaml.load(user_input)"
        ],
        "fix": "Avoid deserializing untrusted data. Use safe alternatives like yaml.safe_load(). Validate and sanitize before deserializing.",
        "beginner_explanation": "Deserialization converts stored data back into objects. If an attacker crafts malicious data, converting it could run harmful code. Use safe loaders and never deserialize untrusted input.",
        "cwe": "CWE-502",
        "owasp": "A08:2021 Software and Data Integrity Failures"
    },
    {
        "id": "path_traversal",
        "name": "Path Traversal",
        "severity": "High",
        "description": "Attackers can access files outside the intended directory using '../' sequences.",
        "patterns": ["open(", "os.path.join", "file_path +", "readFile", "fs.read", "../", "..\\\\"],
        "examples": [
            "open('/files/' + user_input)",
            "file_path = base_dir + filename"
        ],
        "fix": "Validate and sanitize file paths. Use os.path.basename() to extract filename. Check resolved path is within allowed directory.",
        "beginner_explanation": "If a user requests '../../etc/passwd' instead of 'document.pdf', they might access system files. Always validate file paths and remove directory traversal characters.",
        "cwe": "CWE-22",
        "owasp": "A01:2021 Broken Access Control"
    },
    {
        "id": "command_injection",
        "name": "Command Injection",
        "severity": "Critical",
        "description": "Executing system commands with unsanitized user input allows arbitrary command execution.",
        "patterns": ["os.system", "subprocess.call", "subprocess.run", "shell=True", "exec(", "eval(", "child_process", "system("],
        "examples": [
            "os.system('ping ' + user_ip)",
            "subprocess.run(f'convert {user_file}', shell=True)"
        ],
        "fix": "Avoid shell=True. Pass arguments as a list. Validate and whitelist allowed inputs. Use shlex.quote() for escaping.",
        "beginner_explanation": "Running system commands with user input is dangerous. Someone could input '; rm -rf /' and delete everything. Always validate input and use safe subprocess calls.",
        "cwe": "CWE-78",
        "owasp": "A03:2021 Injection"
    },
    {
        "id": "weak_crypto",
        "name": "Weak Cryptography",
        "severity": "High",
        "description": "Using outdated or weak cryptographic algorithms that can be easily broken.",
        "patterns": ["MD5", "SHA1", "DES", "RC4", "base64", "rot13", "random()", "Math.random"],
        "examples": [
            "hashlib.md5(password.encode())",
            "encrypted = base64.b64encode(secret)"
        ],
        "fix": "Use strong algorithms: SHA-256+, bcrypt/argon2 for passwords, AES-256 for encryption, secrets module for random.",
        "beginner_explanation": "Old encryption methods like MD5 can be cracked quickly. It's like using a simple padlock for a bank vault. Use modern algorithms like bcrypt for passwords.",
        "cwe": "CWE-327",
        "owasp": "A02:2021 Cryptographic Failures"
    },
    {
        "id": "insecure_random",
        "name": "Insecure Randomness",
        "severity": "Medium",
        "description": "Using predictable random number generators for security-sensitive operations.",
        "patterns": ["random.random", "random.randint", "Math.random", "rand()", "srand"],
        "examples": [
            "token = random.randint(100000, 999999)",
            "session_id = str(random.random())"
        ],
        "fix": "Use cryptographically secure random: secrets module in Python, crypto.randomBytes in Node.js.",
        "beginner_explanation": "Regular random numbers follow patterns that hackers can predict. For security tokens, use 'secrets' module which generates truly unpredictable numbers.",
        "cwe": "CWE-338",
        "owasp": "A02:2021 Cryptographic Failures"
    }
]

# Build TF-IDF vectorizer for RAG
def build_vulnerability_index():
    """Build TF-IDF index for vulnerability knowledge base."""
    documents = []
    for vuln in VULNERABILITY_KB:
        doc = f"{vuln['name']} {vuln['description']} {' '.join(vuln['patterns'])} {' '.join(vuln['examples'])}"
        documents.append(doc)
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

# Initialize RAG index
try:
    vuln_vectorizer, vuln_tfidf_matrix = build_vulnerability_index()
except Exception:
    vuln_vectorizer, vuln_tfidf_matrix = None, None


def search_vulnerabilities(code: str, top_k: int = 5) -> List[dict]:
    """Search vulnerability KB using TF-IDF similarity."""
    if vuln_vectorizer is None:
        return VULNERABILITY_KB[:top_k]
    
    code_vector = vuln_vectorizer.transform([code])
    similarities = cosine_similarity(code_vector, vuln_tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.05:  # Minimum similarity threshold
            vuln = VULNERABILITY_KB[idx].copy()
            vuln['relevance_score'] = float(similarities[idx])
            results.append(vuln)
    
    return results

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


class MultiAgentRequest(BaseModel):
    code: str
    language: str = "python"


class GitHubScanRequest(BaseModel):
    repo_url: str
    branch: str = "main"


class GenerateReportRequest(BaseModel):
    repo_name: str
    files_analyzed: int
    scores: dict
    agent_results: dict
    summary: str


class RiskPredictionRequest(BaseModel):
    code: str
    language: str = "python"


class EnterpriseReportRequest(BaseModel):
    code: str
    language: str = "python"
    project_name: str = "Code Analysis"
    include_sections: List[str] = ["security", "compliance", "issues", "recommendations"]


class CodeDiffRequest(BaseModel):
    original_code: str
    rewritten_code: str


# AI Agent Configurations
AI_AGENTS = {
    "security": {
        "name": "Security Agent",
        "icon": "🛡️",
        "color": "#ef4444",
        "prompt": """You are a SECURITY EXPERT agent. Analyze this {language} code ONLY for security vulnerabilities.

Focus on:
- SQL injection, XSS, CSRF vulnerabilities
- Authentication/authorization flaws
- Input validation issues
- Sensitive data exposure
- Insecure dependencies
- Hardcoded secrets/credentials
- Buffer overflows or memory issues

Code to analyze:
```{language}
{code}
```

Provide a focused security review in markdown format with:
1. Security Risk Level (Critical/High/Medium/Low)
2. Specific vulnerabilities found
3. Recommended fixes with code examples"""
    },
    "performance": {
        "name": "Performance Agent",
        "icon": "⚡",
        "color": "#f59e0b",
        "prompt": """You are a PERFORMANCE OPTIMIZATION agent. Analyze this {language} code ONLY for performance issues.

Focus on:
- Time complexity issues (O(n²) loops, etc.)
- Memory leaks and inefficient memory usage
- Unnecessary computations or redundant operations
- Database query optimization (N+1 queries, missing indexes)
- Caching opportunities
- Async/parallel processing opportunities
- Resource management

Code to analyze:
```{language}
{code}
```

Provide a focused performance review in markdown format with:
1. Performance Impact Level (Critical/High/Medium/Low)
2. Specific bottlenecks found
3. Optimized code examples"""
    },
    "clean_code": {
        "name": "Clean Code Agent",
        "icon": "🧹",
        "color": "#22c55e",
        "prompt": """You are a CLEAN CODE expert agent. Analyze this {language} code ONLY for code quality and maintainability.

Focus on:
- Code readability and clarity
- Naming conventions (variables, functions, classes)
- Function/method length and complexity
- DRY principle violations (repeated code)
- SOLID principles adherence
- Code comments and documentation
- Error handling patterns
- Code organization and structure

Code to analyze:
```{language}
{code}
```

Provide a focused clean code review in markdown format with:
1. Code Quality Level (Excellent/Good/Fair/Poor)
2. Specific issues found
3. Refactored code examples"""
    },
    "architecture": {
        "name": "Architecture Agent",
        "icon": "📐",
        "color": "#8b5cf6",
        "prompt": """You are a SOFTWARE ARCHITECTURE expert agent. Analyze this {language} code ONLY for architectural patterns and design.

Focus on:
- Design pattern usage and appropriateness
- Separation of concerns
- Dependency management
- Modularity and coupling
- Scalability considerations
- API design (if applicable)
- Layer architecture adherence
- Extensibility and flexibility

Code to analyze:
```{language}
{code}
```

Provide a focused architecture review in markdown format with:
1. Architecture Quality Level (Excellent/Good/Fair/Poor)
2. Design issues found
3. Architectural recommendations with examples"""
    }
}


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
    """Handle code review requests from the frontend using the Groq API with RAG enhancement."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code input cannot be empty")

    focus = ""
    if request.focus_areas:
        focus = f"\nFocus especially on these areas: {', '.join(request.focus_areas)}"

    # RAG: Search for relevant vulnerabilities
    relevant_vulns = search_vulnerabilities(request.code, top_k=3)
    vuln_context = ""
    if relevant_vulns:
        vuln_context = "\n\n**Relevant Vulnerability Patterns to Check:**\n"
        for v in relevant_vulns:
            vuln_context += f"- {v['name']} ({v['severity']}): {v['description']}\n"

    # Mode-specific instructions
    if request.mode == "beginner":
        mode_instructions = """
IMPORTANT: You are explaining to a BEGINNER/STUDENT. For each issue:
1. Use simple, everyday language (avoid jargon)
2. Explain WHY this is a problem using real-world analogies
3. Show the WRONG code vs CORRECT code side by side
4. Provide step-by-step instructions to fix it
5. Include a "💡 Learn More" tip for each major issue
6. Be encouraging and educational, not critical"""
    else:
        mode_instructions = """
Target audience: Professional developers. Be concise and technical.
Include code examples for fixes. Reference best practices and design patterns."""

    prompt = f"""You are an expert code reviewer. Analyze the following {request.language} code and provide a thorough review.{focus}
{mode_instructions}
{vuln_context}

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
            "vulnerabilities_checked": [v["name"] for v in relevant_vulns],
            "mode": request.mode
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


@app.post("/score")
async def score_code(request: CodeScoreRequest):
    """Generate AI-powered code quality scores (0-100) for the submitted code."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code input cannot be empty")

    prompt = f"""You are an expert code quality analyst. Analyze the following {request.language} code and provide quality scores.

Evaluate the code on these dimensions and provide a score from 0 to 100 for each:

1. **Code Quality Score**: Overall code health considering bugs, logic errors, and correctness
2. **Security Score**: Absence of security vulnerabilities, injection risks, unsafe operations
3. **Maintainability Score**: Code readability, modularity, documentation, naming conventions
4. **Performance Score**: Efficiency, algorithmic complexity, resource usage
5. **Reliability Score**: Error handling, edge cases, robustness

Also provide:
- A brief summary (2-3 sentences) of the code's overall health
- Top 3 recommendations for improvement

Code to analyze:
```{request.language}
{request.code}
```

IMPORTANT: Respond ONLY with valid JSON in this exact format, no other text:
{{
    "code_quality": <number 0-100>,
    "security": <number 0-100>,
    "maintainability": <number 0-100>,
    "performance": <number 0-100>,
    "reliability": <number 0-100>,
    "overall": <number 0-100>,
    "summary": "<brief summary>",
    "recommendations": ["<recommendation 1>", "<recommendation 2>", "<recommendation 3>"]
}}"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            temperature=0.2,
            max_tokens=1024,
            top_p=TOP_P,
        )

        response_text = chat_completion.choices[0].message.content.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            raise ValueError("Could not parse JSON from response")
        
        return {"scores": scores}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse scores: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multi-agent-review")
async def multi_agent_review(request: MultiAgentRequest):
    """Run multiple AI agents to review code from different perspectives."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code input cannot be empty")

    results = {}
    
    for agent_id, agent_config in AI_AGENTS.items():
        prompt = agent_config["prompt"].format(
            language=request.language,
            code=request.code
        )
        
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                top_p=TOP_P,
            )
            
            results[agent_id] = {
                "name": agent_config["name"],
                "icon": agent_config["icon"],
                "color": agent_config["color"],
                "analysis": chat_completion.choices[0].message.content
            }
        except Exception as e:
            results[agent_id] = {
                "name": agent_config["name"],
                "icon": agent_config["icon"],
                "color": agent_config["color"],
                "analysis": f"Agent error: {str(e)}"
            }
    
    return {"agents": results}


@app.get("/logo.png")
async def serve_logo():
    """Serve the logo image."""
    file_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "logo.png")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo not found")


# ---------- GitHub Integration ----------

def parse_github_url(repo_url: str) -> tuple:
    """Parse GitHub URL to extract owner and repo name."""
    # Support formats: 
    # https://github.com/owner/repo
    # https://github.com/owner/repo.git
    # github.com/owner/repo
    repo_url = repo_url.strip().rstrip('/')
    if repo_url.endswith('.git'):
        repo_url = repo_url[:-4]
    
    patterns = [
        r'github\.com/([^/]+)/([^/]+)',
        r'^([^/]+)/([^/]+)$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, repo_url)
        if match:
            return match.group(1), match.group(2)
    
    raise ValueError("Invalid GitHub URL format")


def get_repo_files(owner: str, repo: str, branch: str = "main", path: str = "") -> List[dict]:
    """Fetch repository files using GitHub API."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": branch}
    
    response = requests.get(api_url, params=params, timeout=30)
    
    if response.status_code == 404:
        # Try 'master' branch if 'main' fails
        if branch == "main":
            params["ref"] = "master"
            response = requests.get(api_url, params=params, timeout=30)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, 
                          detail=f"GitHub API error: {response.json().get('message', 'Unknown error')}")
    
    return response.json()


def get_file_content(url: str) -> str:
    """Fetch file content from GitHub raw URL."""
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        return response.text
    return ""


def is_code_file(filename: str) -> bool:
    """Check if file is a code file based on extension."""
    code_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
        '.html', '.css', '.scss', '.vue', '.svelte', '.json', '.yaml', '.yml'
    }
    ext = os.path.splitext(filename)[1].lower()
    return ext in code_extensions


def get_language_from_extension(filename: str) -> str:
    """Get programming language from file extension."""
    ext_map = {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
        '.jsx': 'javascript', '.tsx': 'typescript', '.java': 'java',
        '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.cs': 'csharp',
        '.go': 'go', '.rs': 'rust', '.rb': 'ruby', '.php': 'php',
        '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala',
        '.html': 'html', '.css': 'css', '.vue': 'vue', '.json': 'json'
    }
    ext = os.path.splitext(filename)[1].lower()
    return ext_map.get(ext, 'text')


@app.post("/github-scan")
async def scan_github_repo(request: GitHubScanRequest):
    """Scan a GitHub repository and analyze all code files."""
    try:
        owner, repo = parse_github_url(request.repo_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Fetch repository structure
    files_to_analyze = []
    
    def fetch_files_recursive(path: str = ""):
        try:
            items = get_repo_files(owner, repo, request.branch, path)
            if not isinstance(items, list):
                items = [items]
            
            for item in items:
                if item["type"] == "file" and is_code_file(item["name"]):
                    if item.get("size", 0) < 100000:  # Skip files > 100KB
                        files_to_analyze.append({
                            "name": item["name"],
                            "path": item["path"],
                            "download_url": item.get("download_url"),
                            "size": item.get("size", 0)
                        })
                elif item["type"] == "dir" and not item["name"].startswith('.'):
                    # Skip common non-code directories
                    skip_dirs = {'node_modules', 'venv', '.venv', '__pycache__', 'dist', 'build', '.git'}
                    if item["name"] not in skip_dirs:
                        fetch_files_recursive(item["path"])
        except Exception:
            pass
    
    fetch_files_recursive()
    
    if not files_to_analyze:
        raise HTTPException(status_code=404, detail="No code files found in repository")
    
    # Limit files to analyze (max 10 for performance)
    files_to_analyze = files_to_analyze[:10]
    
    # Fetch content and analyze
    all_code = ""
    file_details = []
    
    for file_info in files_to_analyze:
        if file_info["download_url"]:
            content = get_file_content(file_info["download_url"])
            if content:
                language = get_language_from_extension(file_info["name"])
                file_details.append({
                    "name": file_info["name"],
                    "path": file_info["path"],
                    "language": language,
                    "lines": len(content.split('\n')),
                    "size": file_info["size"]
                })
                all_code += f"\n\n# === File: {file_info['path']} ===\n{content}"
    
    # Generate overall project analysis using AI
    project_prompt = f"""You are an expert code reviewer analyzing a GitHub repository.

Repository: {owner}/{repo}
Files analyzed: {len(file_details)}

Combined code from the repository:
{all_code[:15000]}  # Limit to 15K chars

Provide a comprehensive project analysis in JSON format:
{{
    "project_summary": "<2-3 sentence overview of what this project does>",
    "tech_stack": ["<list of technologies/frameworks detected>"],
    "code_quality": <score 0-100>,
    "security_score": <score 0-100>,
    "maintainability_score": <score 0-100>,
    "performance_score": <score 0-100>,
    "overall_score": <score 0-100>,
    "critical_issues": ["<list of critical issues found>"],
    "recommendations": ["<top 5 recommendations>"],
    "strengths": ["<list of good practices found>"]
}}

Respond ONLY with valid JSON."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": project_prompt}],
            model=MODEL_NAME,
            temperature=0.2,
            max_tokens=2048,
            top_p=TOP_P,
        )
        
        response_text = chat_completion.choices[0].message.content.strip()
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            analysis = {"error": "Could not parse analysis"}
            
    except Exception as e:
        analysis = {"error": str(e)}
    
    return {
        "repo_name": f"{owner}/{repo}",
        "branch": request.branch,
        "files_analyzed": len(file_details),
        "files": file_details,
        "analysis": analysis
    }


@app.post("/generate-report")
async def generate_pdf_report(request: GenerateReportRequest):
    """Generate a downloadable PDF report from analysis results."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                           rightMargin=50, leftMargin=50,
                           topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4F46E5')
    )
    
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#1E293B')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        alignment=TA_JUSTIFY
    )
    
    story = []
    
    # Title
    story.append(Paragraph("🔍 CodeRefine AI Report", title_style))
    story.append(Paragraph(f"Repository: {request.repo_name}", styles['Heading3']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Summary section
    story.append(Paragraph("📋 Executive Summary", header_style))
    story.append(Paragraph(request.summary or "No summary available.", body_style))
    story.append(Spacer(1, 10))
    
    # Scores table
    story.append(Paragraph("📊 Quality Scores", header_style))
    
    scores = request.scores
    score_data = [
        ['Metric', 'Score', 'Rating'],
        ['Overall', str(scores.get('overall_score', 'N/A')), get_rating(scores.get('overall_score', 0))],
        ['Code Quality', str(scores.get('code_quality', 'N/A')), get_rating(scores.get('code_quality', 0))],
        ['Security', str(scores.get('security_score', 'N/A')), get_rating(scores.get('security_score', 0))],
        ['Maintainability', str(scores.get('maintainability_score', 'N/A')), get_rating(scores.get('maintainability_score', 0))],
        ['Performance', str(scores.get('performance_score', 'N/A')), get_rating(scores.get('performance_score', 0))],
    ]
    
    score_table = Table(score_data, colWidths=[2.5*inch, 1*inch, 1.5*inch])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F46E5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F5F9')]),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 20))
    
    # Critical Issues
    story.append(Paragraph("⚠️ Critical Issues", header_style))
    critical_issues = scores.get('critical_issues', [])
    if critical_issues:
        for issue in critical_issues[:5]:
            story.append(Paragraph(f"• {issue}", body_style))
    else:
        story.append(Paragraph("No critical issues found.", body_style))
    story.append(Spacer(1, 10))
    
    # Recommendations
    story.append(Paragraph("💡 Recommendations", header_style))
    recommendations = scores.get('recommendations', [])
    if recommendations:
        for rec in recommendations[:5]:
            story.append(Paragraph(f"• {rec}", body_style))
    else:
        story.append(Paragraph("No specific recommendations.", body_style))
    story.append(Spacer(1, 10))
    
    # Strengths
    story.append(Paragraph("✅ Strengths", header_style))
    strengths = scores.get('strengths', [])
    if strengths:
        for strength in strengths[:5]:
            story.append(Paragraph(f"• {strength}", body_style))
    else:
        story.append(Paragraph("Analysis pending.", body_style))
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph("─" * 50, styles['Normal']))
    story.append(Paragraph("Generated by CodeRefine AI • Powered by Llama 3.3 70B", 
                          ParagraphStyle('Footer', alignment=TA_CENTER, fontSize=9, textColor=colors.gray)))
    
    doc.build(story)
    buffer.seek(0)
    
    filename = f"CodeRefine_Report_{request.repo_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


def get_rating(score: int) -> str:
    """Convert score to rating text."""
    if not isinstance(score, (int, float)):
        return "N/A"
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 50:
        return "Average"
    if score >= 30:
        return "Poor"
    return "Critical"


# ============ AI RISK PREDICTION DASHBOARD ============

@app.post("/risk-prediction")
async def predict_risks(request: RiskPredictionRequest):
    """AI-powered risk prediction for code - runtime failure, security breach, maintainability."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code input cannot be empty")

    # Search for relevant vulnerabilities using RAG
    relevant_vulns = search_vulnerabilities(request.code, top_k=5)
    vuln_context = "\n".join([f"- {v['name']}: {v['description']}" for v in relevant_vulns])

    prompt = f"""You are an AI risk analyst for software. Analyze this {request.language} code and predict potential risks.

**Relevant Vulnerability Patterns Detected:**
{vuln_context}

Code to analyze:
```{request.language}
{request.code}
```

Provide a comprehensive risk assessment in this exact JSON format:
{{
    "runtime_failure_risk": {{
        "probability": <0-100>,
        "level": "<Critical/High/Medium/Low>",
        "factors": ["<list of factors contributing to runtime risk>"],
        "mitigation": ["<list of mitigation steps>"]
    }},
    "security_breach_risk": {{
        "probability": <0-100>,
        "level": "<Critical/High/Medium/Low>",
        "vulnerabilities": ["<list of specific vulnerabilities>"],
        "attack_vectors": ["<potential attack methods>"],
        "mitigation": ["<list of security measures>"]
    }},
    "maintainability_risk": {{
        "score": <0-100 where 100 is best>,
        "level": "<Excellent/Good/Fair/Poor>",
        "issues": ["<list of maintainability concerns>"],
        "improvements": ["<list of suggested improvements>"]
    }},
    "technical_debt": {{
        "score": <0-100 where 0 is no debt>,
        "hours_to_fix": <estimated hours>,
        "priority_fixes": ["<top 3 fixes to reduce debt>"]
    }},
    "overall_risk_score": <0-100 where 0 is safest>,
    "risk_summary": "<2-3 sentence summary of overall risk posture>"
}}

Respond ONLY with valid JSON."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            temperature=0.2,
            max_tokens=2048,
            top_p=TOP_P,
        )

        response_text = chat_completion.choices[0].message.content.strip()
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            risks = json.loads(json_match.group())
        else:
            raise ValueError("Could not parse JSON from response")

        return {
            "risks": risks,
            "vulnerabilities_detected": [v["name"] for v in relevant_vulns]
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse risk assessment: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ CODE DIFF VISUALIZER ============

def generate_diff(original: str, rewritten: str) -> List[dict]:
    """Generate line-by-line diff between original and rewritten code."""
    import difflib
    
    original_lines = original.splitlines(keepends=True)
    rewritten_lines = rewritten.splitlines(keepends=True)
    
    differ = difflib.unified_diff(original_lines, rewritten_lines, lineterm='')
    
    diff_result = []
    line_num_old = 0
    line_num_new = 0
    
    for line in differ:
        if line.startswith('@@'):
            # Parse hunk header
            match = re.search(r'@@ -(\d+)', line)
            if match:
                line_num_old = int(match.group(1)) - 1
                line_num_new = int(match.group(1)) - 1
            diff_result.append({"type": "info", "content": line.strip(), "old_line": None, "new_line": None})
        elif line.startswith('---') or line.startswith('+++'):
            continue
        elif line.startswith('-'):
            line_num_old += 1
            diff_result.append({
                "type": "removed",
                "content": line[1:].rstrip('\n'),
                "old_line": line_num_old,
                "new_line": None
            })
        elif line.startswith('+'):
            line_num_new += 1
            diff_result.append({
                "type": "added",
                "content": line[1:].rstrip('\n'),
                "old_line": None,
                "new_line": line_num_new
            })
        else:
            line_num_old += 1
            line_num_new += 1
            diff_result.append({
                "type": "unchanged",
                "content": line[1:].rstrip('\n') if line.startswith(' ') else line.rstrip('\n'),
                "old_line": line_num_old,
                "new_line": line_num_new
            })
    
    return diff_result


@app.post("/code-diff")
async def get_code_diff(request: CodeDiffRequest):
    """Generate visual diff between original and rewritten code."""
    if not request.original_code.strip() or not request.rewritten_code.strip():
        raise HTTPException(status_code=400, detail="Both original and rewritten code are required")
    
    diff = generate_diff(request.original_code, request.rewritten_code)
    
    # Calculate statistics
    added = sum(1 for d in diff if d["type"] == "added")
    removed = sum(1 for d in diff if d["type"] == "removed")
    unchanged = sum(1 for d in diff if d["type"] == "unchanged")
    
    return {
        "diff": diff,
        "stats": {
            "lines_added": added,
            "lines_removed": removed,
            "lines_unchanged": unchanged,
            "total_changes": added + removed,
            "change_percentage": round((added + removed) / max(len(request.original_code.splitlines()), 1) * 100, 1)
        }
    }


# ============ VULNERABILITY KNOWLEDGE BASE ENDPOINT ============

@app.get("/vulnerability-kb")
async def get_vulnerability_kb():
    """Return the vulnerability knowledge base for reference."""
    return {
        "vulnerabilities": VULNERABILITY_KB,
        "total": len(VULNERABILITY_KB)
    }


@app.post("/search-vulnerabilities")
async def search_vuln_kb(code: str):
    """Search vulnerability KB for patterns matching the code."""
    results = search_vulnerabilities(code, top_k=5)
    return {
        "matches": results,
        "total_matches": len(results)
    }


# ============ ENTERPRISE REPORTING MODULE ============

@app.post("/enterprise-report")
async def generate_enterprise_report(request: EnterpriseReportRequest):
    """Generate comprehensive enterprise-grade PDF report with security compliance."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code input cannot be empty")

    # Step 1: Get all analyses
    # Security scan
    security_prompt = f"""Analyze this {request.language} code for security compliance and generate a security audit.

Code:
```{request.language}
{request.code}
```

Provide analysis in JSON format:
{{
    "security_grade": "<A/B/C/D/F>",
    "compliance_status": {{
        "OWASP_Top_10": "<Compliant/Partial/Non-Compliant>",
        "CWE_Top_25": "<Compliant/Partial/Non-Compliant>",
        "SANS_Top_25": "<Compliant/Partial/Non-Compliant>"
    }},
    "vulnerabilities": [
        {{"id": "VULN-001", "severity": "Critical/High/Medium/Low", "title": "", "description": "", "line": 0, "recommendation": ""}}
    ],
    "security_findings": {{
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "informational": 0
    }},
    "executive_summary": "<2-3 sentence security posture summary>"
}}

Respond ONLY with valid JSON."""

    try:
        # Get security analysis
        security_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": security_prompt}],
            model=MODEL_NAME,
            temperature=0.2,
            max_tokens=2048,
        )
        security_text = security_completion.choices[0].message.content.strip()
        security_match = re.search(r'\{[\s\S]*\}', security_text)
        security_data = json.loads(security_match.group()) if security_match else {}

    except Exception as e:
        security_data = {"error": str(e)}

    # Get vulnerability KB matches
    vuln_matches = search_vulnerabilities(request.code, top_k=5)

    # Generate PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                           rightMargin=40, leftMargin=40,
                           topMargin=40, bottomMargin=40)

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=28, spaceAfter=20, alignment=TA_CENTER, textColor=colors.HexColor('#1a1a2e'))
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=12, spaceAfter=30, alignment=TA_CENTER, textColor=colors.gray)
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=16, spaceBefore=25, spaceAfter=12, textColor=colors.HexColor('#4F46E5'))
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, spaceAfter=8, alignment=TA_JUSTIFY)

    story = []

    # Cover Page
    story.append(Spacer(1, 100))
    story.append(Paragraph("🔐 ENTERPRISE SECURITY AUDIT", title_style))
    story.append(Paragraph(f"Project: {request.project_name}", styles['Heading3']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}", subtitle_style))
    story.append(Paragraph(f"Language: {request.language.upper()}", styles['Normal']))
    story.append(Spacer(1, 50))

    # Security Grade Badge
    grade = security_data.get('security_grade', 'N/A')
    grade_color = {'A': '#22c55e', 'B': '#84cc16', 'C': '#eab308', 'D': '#f97316', 'F': '#ef4444'}.get(grade, '#6b7280')
    story.append(Paragraph(f"<font color='{grade_color}' size='48'><b>GRADE: {grade}</b></font>", ParagraphStyle('Grade', alignment=TA_CENTER)))
    story.append(PageBreak())

    # Executive Summary
    story.append(Paragraph("📋 Executive Summary", section_style))
    story.append(Paragraph(security_data.get('executive_summary', 'Security analysis completed.'), body_style))
    story.append(Spacer(1, 15))

    # Compliance Status
    story.append(Paragraph("✅ Compliance Status", section_style))
    compliance = security_data.get('compliance_status', {})
    compliance_data = [
        ['Standard', 'Status'],
        ['OWASP Top 10 2021', compliance.get('OWASP_Top_10', 'N/A')],
        ['CWE Top 25', compliance.get('CWE_Top_25', 'N/A')],
        ['SANS Top 25', compliance.get('SANS_Top_25', 'N/A')],
    ]
    compliance_table = Table(compliance_data, colWidths=[3*inch, 2*inch])
    compliance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F46E5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(compliance_table)
    story.append(Spacer(1, 20))

    # Findings Summary
    story.append(Paragraph("📊 Security Findings Summary", section_style))
    findings = security_data.get('security_findings', {})
    findings_data = [
        ['Severity', 'Count'],
        ['🔴 Critical', str(findings.get('critical', 0))],
        ['🟠 High', str(findings.get('high', 0))],
        ['🟡 Medium', str(findings.get('medium', 0))],
        ['🟢 Low', str(findings.get('low', 0))],
        ['🔵 Informational', str(findings.get('informational', 0))],
    ]
    findings_table = Table(findings_data, colWidths=[2.5*inch, 1.5*inch])
    findings_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 20))

    # Vulnerability Details
    story.append(Paragraph("🔍 Vulnerability Details", section_style))
    vulns = security_data.get('vulnerabilities', [])
    if vulns:
        for v in vulns[:10]:  # Limit to 10
            story.append(Paragraph(f"<b>{v.get('id', 'N/A')}</b> - {v.get('title', 'Issue')}", body_style))
            story.append(Paragraph(f"Severity: {v.get('severity', 'N/A')} | Line: {v.get('line', 'N/A')}", ParagraphStyle('Small', fontSize=9, textColor=colors.gray)))
            story.append(Paragraph(v.get('description', ''), body_style))
            story.append(Paragraph(f"<b>Recommendation:</b> {v.get('recommendation', 'N/A')}", body_style))
            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("No critical vulnerabilities detected.", body_style))

    # Knowledge Base Matches
    story.append(PageBreak())
    story.append(Paragraph("📚 Vulnerability Knowledge Base Matches", section_style))
    if vuln_matches:
        for v in vuln_matches:
            story.append(Paragraph(f"<b>{v['name']}</b> ({v['severity']})", body_style))
            story.append(Paragraph(f"CWE: {v.get('cwe', 'N/A')} | OWASP: {v.get('owasp', 'N/A')}", ParagraphStyle('Small', fontSize=9, textColor=colors.gray)))
            story.append(Paragraph(v['description'], body_style))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No matching vulnerability patterns found.", body_style))

    # Footer
    story.append(Spacer(1, 40))
    story.append(Paragraph("─" * 60, styles['Normal']))
    story.append(Paragraph("Generated by CodeRefine AI Enterprise • Powered by Llama 3.3 70B + RAG",
                          ParagraphStyle('Footer', alignment=TA_CENTER, fontSize=8, textColor=colors.gray)))
    story.append(Paragraph("This report is for informational purposes. Professional security audit recommended.",
                          ParagraphStyle('Disclaimer', alignment=TA_CENTER, fontSize=7, textColor=colors.gray)))

    doc.build(story)
    buffer.seek(0)

    filename = f"Enterprise_Security_Audit_{request.project_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)