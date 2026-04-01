from flask import Flask, request, jsonify, send_from_directory, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from dotenv import load_dotenv
import json
import os
import requests
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables from .env file
load_dotenv()

from db import create_tables, add_user, get_user, store_reset_token, verify_reset_token, update_password

# -------------------------------
# Initialize
# -------------------------------
create_tables()  # Create DB tables if needed

app = Flask(__name__)
CORS(app)

# -------------------------------
# API Keys (loaded from .env)
# -------------------------------
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "").strip('"')
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "").strip('"')
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "").strip('"')
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "").strip('"')

# Email Configuration
EMAIL_USER = os.getenv("EMAIL_USER", "").strip('"')
EMAIL_PASS = os.getenv("EMAIL_PASS", "").strip('"')

# -------------------------------
# Load careers datasets
# -------------------------------
# Full mapping (subjects to courses)
with open("kuccps_structured_full.json") as f:
    careers_data = json.load(f)

# Flat list with cutoffs
with open("kuccps_structured.json") as f:
    cutoff_data = json.load(f)

# Create a lookup map for faster cutoff access (cleaned names)
cutoff_map = {item["course"].lower().strip(): item["cutoff"] for item in cutoff_data}

def get_cutoff(course_name):
    """Smarter cut-off lookup with common aliases and fallbacks."""
    c = course_name.lower().strip()
    
    # Common Alias Mapping
    aliases = {
        "ict": "information and communication technology",
        "it": "information technology",
        "cs": "computer science",
        "hr": "human resource",
    }
    for short, long in aliases.items():
        if short in c: c = c.replace(short, long)

    # 1. Direct match
    if c in cutoff_map: return cutoff_map[c]
    
    # 2. Word-weighted matching (for higher accuracy than simple 'in')
    best_match = "N/A"
    max_overlap = 0
    c_words = set(c.split())
    
    for name, val in cutoff_map.items():
        name_words = set(name.split())
        overlap = len(c_words.intersection(name_words))
        if overlap > max_overlap and overlap >= 2:
            max_overlap = overlap
            best_match = val
            
    # 3. Default range fallbacks if still N/A
    if best_match == "N/A":
        if "diploma" in c: return "15.68 (Est.)"
        if "certificate" in c: return "10.00 (Est.)"
        
    return best_match


# -------------------------------
# Provider functions
# -------------------------------

# Groq models to try (fast, free tier, OpenAI-compatible)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

def try_groq(prompt):
    """PRIMARY: Groq — ultra-fast inference, OpenAI-compatible, free tier."""
    if not GROQ_API_KEY:
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    for model in GROQ_MODELS:
        try:
            resp = requests.post(url, headers=headers, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
            }, timeout=20)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                if content and content.strip():
                    print(f"[Groq] OK: {model}")
                    return content
            else:
                print(f"[Groq] {model} -> {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"[Groq] {model} -> Error: {e}")
    return None



# Free models to try on OpenRouter (tries each until one works)
OPENROUTER_FREE_MODELS = [
    "openai/gpt-oss-20b:free",
    "openai/gpt-oss-120b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "liquid/lfm-7b:free",
    "google/gemini-2.0-flash-exp:free",
]

def try_openrouter(prompt):
    """PRIMARY: Try free OpenRouter models until one succeeds."""
    if not OPENROUTER_API_KEY:
        return None
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Career Recommendation System",
    }
    for model in OPENROUTER_FREE_MODELS:
        try:
            resp = requests.post(url, headers=headers, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }, timeout=30)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                if content and content.strip():
                    print(f"[OpenRouter] OK: {model}")
                    return content
            else:
                print(f"[OpenRouter] {model} -> {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"[OpenRouter] {model} -> Error: {e}")
    return None


def try_huggingface(prompt):
    """FALLBACK 1: HuggingFace Inference Router (new endpoint)."""
    if not HUGGINGFACE_API_KEY:
        return None
    hf_models = [
        "Qwen/Qwen2.5-72B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-3.5-mini-instruct",
    ]
    for model in hf_models:
        url = f"https://router.huggingface.co/hf-inference/models/{model}/v1/chat/completions"
        try:
            resp = requests.post(url, headers={
                "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                "Content-Type": "application/json",
            }, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
            }, timeout=25)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                if content and content.strip():
                    print(f"[HuggingFace] OK: {model}")
                    return content
            else:
                print(f"[HuggingFace] {model} -> {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"[HuggingFace] {model} -> Error: {e}")
    return None


def try_openai(prompt):
    """FALLBACK 2: OpenAI Chat Completions."""
    if not OPENAI_API_KEY:
        return None
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}]},
            timeout=20)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            print(f"[OpenAI] {resp.status_code}: {resp.text[:150]}")
    except Exception as e:
        print(f"[OpenAI] Error: {e}")
    return None


def rule_based_recommendation(skills, interests, education):
    """GUARANTEED FALLBACK: Always returns meaningful career advice based on inputs."""
    career_map = {
        "technology": [
            ("Software Engineer", "Design and build software applications and systems."),
            ("Data Scientist", "Analyse data to help organisations make better decisions."),
            ("Cybersecurity Analyst", "Protect computer systems from digital attacks."),
            ("AI/ML Engineer", "Build machine learning models and intelligent systems."),
            ("Web Developer", "Create websites and web-based applications."),
        ],
        "business": [
            ("Business Analyst", "Identify business needs and recommend solutions."),
            ("Entrepreneur", "Start and run your own business venture."),
            ("Accountant / CPA", "Manage financial records and statements."),
            ("Marketing Manager", "Plan and execute marketing strategies."),
            ("Finance Manager", "Oversee financial planning and budgeting."),
        ],
        "healthcare": [
            ("Medical Doctor", "Diagnose and treat illnesses and injuries."),
            ("Nurse", "Provide patient care and support medical teams."),
            ("Pharmacist", "Dispense medicines and counsel patients."),
            ("Clinical Officer", "Provide medical services in clinical settings."),
            ("Public Health Officer", "Promote community health and prevent disease."),
        ],
        "arts": [
            ("Graphic Designer", "Create visual content for digital and print media."),
            ("Journalist / Media", "Report news and tell stories for the public."),
            ("Teacher / Educator", "Teach and inspire the next generation."),
            ("Architect", "Design buildings and urban spaces."),
            ("Content Creator", "Produce digital content for social platforms."),
        ],
    }
    skill_map = {
        "problem solving": [
            ("Engineer", "Apply science and math to solve real-world problems."),
            ("Research Analyst", "Investigate data and present findings."),
        ],
        "communication": [
            ("Public Relations Officer", "Manage an organisation's public image."),
            ("Teacher", "Educate and communicate knowledge to students."),
        ],
        "creativity": [
            ("UI/UX Designer", "Design user-friendly digital interfaces."),
            ("Creative Director", "Lead creative projects and campaigns."),
        ],
        "leadership": [
            ("Project Manager", "Plan and lead projects to completion."),
            ("Operations Manager", "Oversee day-to-day business operations."),
        ],
    }

    matched = []
    interests_l = (interests or "").lower()
    skills_l    = (skills or "").lower()

    for keyword, careers in career_map.items():
        if keyword in interests_l:
            matched.extend(careers)

    for keyword, careers in skill_map.items():
        if keyword in skills_l:
            matched.extend(careers)

    # Remove duplicates preserving order
    seen = set()
    unique = []
    for c in matched:
        if c[0] not in seen:
            seen.add(c[0])
            unique.append(c)

    if not unique:
        unique = [
            ("Software Engineer", "Build digital products and solutions."),
            ("Business Analyst", "Bridge business needs and technical solutions."),
            ("Teacher / Educator", "Share knowledge and inspire students."),
            ("Accountant", "Manage finances for individuals or organisations."),
            ("Public Health Officer", "Promote health and prevent diseases in communities."),
        ]

    lines = [
        f"Based on your interest in {interests or 'various fields'} and strength in {skills or 'multiple skills'}, here are your recommended career paths:\n"
    ]
    for i, (title, desc) in enumerate(unique[:5], 1):
        lines.append(f"{i}. {title}\n   {desc}")

    lines.append(f"\nEducation level: {education or 'KCSE Student'}")
    lines.append("Tip: Explore these careers further through KUCCPS, university open days, or online platforms like Coursera and edX.")

    print("[INFO] Used: Rule-based fallback")
    return json.dumps([
    {
        "career": title,
        "cutoff": get_cutoff(title),
        "description": desc
    }
    for title, desc in unique[:5]
])


def get_recommendation(prompt, skills="", interests="", education=""):
    """Try Groq → OpenRouter → HuggingFace → OpenAI → Rule-based (always works)."""
    result = try_groq(prompt)
    if result:
        return result

    result = try_openrouter(prompt)
    if result:
        return result

    result = try_huggingface(prompt)
    if result:
        return result

    result = try_openai(prompt)
    if result:
        return result

    # Guaranteed fallback — never shows "undefined" or blank
    return rule_based_recommendation(skills, interests, education)


# -------------------------------
# Routes
# -------------------------------

# Serve frontend
@app.route("/")
def serve_index():
    resp = make_response(send_from_directory('.', 'index.html'))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp

@app.route("/<path:path>")
def serve_files(path):
    resp = make_response(send_from_directory('.', path))
    if path.endswith('.html'):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data    = request.json
    profile = data.get("profile", {})
    grades  = profile.get("grades", {})
    prefs   = profile.get("preferences", {})
    
    skills    = prefs.get("skills", "")
    interests = prefs.get("interests", "")
    workstyle = prefs.get("workstyle", "")
    education = data.get("education", "KCSE Student")

    # Calculate Mean Points for context
    grade_to_points = {"A":12,"A-":11,"B+":10,"B":9,"B-":8,"C+":7,"C":6,"C-":5,"D+":4,"D":3,"D-":2,"E":1}
    pts = [grade_to_points.get(v, 1) for v in grades.values() if v and v != "-- Select Grade --"]
    mean_pts = sum(pts)/7 if len(pts)>=7 else (sum(pts)/len(pts) if pts else 1)

    # Determine course level based on mean points
    # KUCCPS levels: Degree (C+ and above), Diploma (C- and above), Craft/Cert (below D+)
    level = "degree"
    if mean_pts < 6.5: level = "diploma"
    if mean_pts < 4.5: level = "certificate"

    # Extract relevant courses from the full dataset for context
    context_courses = []
    keywords = (skills + " " + interests).lower()
    
    # Simple category picker
    cat_to_use = "math" # Default
    if any(k in keywords for k in ["bio", "health", "medicine", "nurse", "science"]):
        cat_to_use = "bio"
    elif any(k in keywords for k in ["business", "commerce", "accounting", "marketing"]):
        cat_to_use = "business"
    
    # Get courses for THIS level and their cutoffs
    raw_list = careers_data.get(cat_to_use, {}).get(level, [])[:15]
    for c_name in raw_list:
        context_courses.append({
            "name": c_name,
            "cutoff": get_cutoff(c_name)
        })

    prompt = (
        f"You are a professional career counsellor. Recommend exactly 5 specific career paths "
        f"for a Kenyan KCSE student with a **Mean Academic Score of {mean_pts:.2f}**.\n\n"
        f"ACADEMIC PROFILE (Subject: Grade):\n"
        f"{json.dumps(grades, indent=2)}\n\n"
        f"PERSONAL PREFERENCES:\n"
        f"- Main Interest: {interests}\n"
        f"- Strongest Skill: {skills}\n"
        f"- Preferred Work Style: {workstyle}\n\n"
        f"AVAILABLE REAL KUCCPS {level.upper()} DATA (Name & Cut-off):\n"
        f"{json.dumps(context_courses, indent=2)}\n\n"
        f"CRITICAL RULES:\n"
        f"1. ONLY recommend {level.upper()} courses. If the student has {mean_pts:.2f} mean points, DO NOT recommend Degree courses unless they meet the cut-off.\n"
        f"2. Use the **EXACT NAMES** provided in the 'AVAILABLE REAL KUCCPS DATA' list (NO ABBREVIATIONS like ICT or IT).\n"
        f"3. Ensure the recommendation matches their 'Main Interest' ({interests}) and 'Work Style' ({workstyle}).\n"
        f"4. Return ONLY a JSON array of objects with keys: 'career', 'cutoff', 'description'.\n"
        f"5. 'description' must be under 12 words, plain text (no markdown, no **).\n"
        f"6. NO PREAMBLE OR EXPLANATION. JUST THE JSON ARRAY."
    )

    recommendation = get_recommendation(prompt, skills, interests, education)
    return jsonify({"result": recommendation})


# Get all careers
@app.route("/api/careers", methods=["GET"])
def get_careers():
    return jsonify(careers_data)


# User Registration
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    first_name = data.get("first_name")
    surname    = data.get("surname")
    email      = data.get("email")
    password   = data.get("password")
    education  = data.get("education")

    if not all([first_name, surname, email, password, education]):
        return jsonify({"error": "All fields are required"}), 400

    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long"}), 400

    hashed_pw = generate_password_hash(password)
    if add_user(first_name, surname, email, hashed_pw, education):
        return jsonify({"message": "Registration successful"})
    return jsonify({"error": "Registration failed (email might already exist)"}), 400


# User Login
@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    email    = data.get("email")
    password = data.get("password")

    user = get_user(email)
    if user and check_password_hash(user["password"], password):
        return jsonify({
            "message": "Login successful",
            "user": {
                "first_name": user["first_name"],
                "surname": user["surname"],
                "email": user["email"],
                "education": user["education"]
            }
        })
    return jsonify({"error": "Invalid email or password"}), 401
# Forgot Password setup
def send_reset_email(to_email, otp):
    if not EMAIL_USER or not EMAIL_PASS or EMAIL_USER == "your-email@gmail.com":
        print("[WARNING] Email credentials not configured in .env. Cannot send real email.")
        return False
        
    try:
        msg = MIMEMultipart()
        msg['From'] = f"Career Path <{EMAIL_USER}>"
        msg['To'] = to_email
        msg['Subject'] = "Your Password Reset Code"
        
        body = f"Hello,\n\nYour password reset code is: {otp}\n\nThis code is valid for your account.\nIf you did not request this, please ignore this email.\n\nBest,\nCareer Path Team"
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False

@app.route("/api/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json()
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email is required"}), 400
    
    user = get_user(email)
    if not user:
        return jsonify({"message": "If an account exists, a reset code has been sent."})
    
    otp = ''.join(random.choices(string.digits, k=6))
    
    if store_reset_token(email, otp):
        send_reset_email(email, otp)
        print("\n" + "="*50)
        print(f"🔐 RESET CODE GENERATED FOR {email}: {otp}")
        print("="*50 + "\n")
        return jsonify({
            "message": "If an account exists, a reset code has been sent to your email."
        })
    return jsonify({"error": "Failed to generate reset code"}), 500

@app.route("/api/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json()
    email = data.get("email")
    token = data.get("token")
    new_password = data.get("new_password")
    
    if not all([email, token, new_password]):
        return jsonify({"error": "Email, token, and new password are required"}), 400
        
    if len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long"}), 400

    if verify_reset_token(email, token):
        hashed_pw = generate_password_hash(new_password)
        if update_password(email, hashed_pw):
            return jsonify({"message": "Password updated successfully!"})
        return jsonify({"error": "Failed to update password"}), 500
        
    return jsonify({"error": "Invalid or expired reset code"}), 400


# Error handling
@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error"}), 500


# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    print("=== API Key Status ===")
    print(f"  Groq        : {'[OK]' if GROQ_API_KEY        else '[MISSING]'}")
    print(f"  HuggingFace : {'[OK]' if HUGGINGFACE_API_KEY else '[MISSING]'}")
    print(f"  OpenRouter  : {'[OK]' if OPENROUTER_API_KEY  else '[MISSING]'}")
    print(f"  OpenAI      : {'[OK]' if OPENAI_API_KEY      else '[MISSING]'}")
    print("======================")
    app.run(debug=True)
