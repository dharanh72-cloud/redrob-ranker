"""
config.py — Redrob Hackathon Ranker Configuration
All JD constants, scoring weights, keyword lists, and disqualifier rules.
Change values here; never hardcode them inside scorer modules.
"""

from __future__ import annotations
from datetime import date

# ─────────────────────────────────────────────────────────────────────────────
# SCORING COMPONENT WEIGHTS  (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────────────
WEIGHTS: dict[str, float] = {
    "skills":   0.30,   # Skill match quality (depth + verification)
    "career":   0.25,   # Career substance (product cos, shipped systems)
    "signals":  0.20,   # Behavioral availability signals
    "experience": 0.15, # Years-of-experience fit to JD band
    "location": 0.10,   # Location + logistics fit
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ─────────────────────────────────────────────────────────────────────────────
# JD: EXPERIENCE BAND
# ─────────────────────────────────────────────────────────────────────────────
YOE_IDEAL_MIN    = 6.0   # Sweet spot lower bound (JD says 6-8)
YOE_IDEAL_MAX    = 8.0   # Sweet spot upper bound
YOE_ACCEPT_MIN   = 4.0   # Hard floor — below this: near-zero score
YOE_ACCEPT_MAX   = 12.0  # Soft ceiling — above this: diminishing returns
YOE_HARD_MIN     = 3.0   # Absolute minimum — below this: score = 0


# ─────────────────────────────────────────────────────────────────────────────
# JD: LOCATION
# ─────────────────────────────────────────────────────────────────────────────
PREFERRED_CITIES: set[str] = {
    "pune", "noida",
}

ACCEPTABLE_CITIES: set[str] = {
    "hyderabad", "mumbai", "delhi", "bengaluru", "bangalore",
    "gurgaon", "gurugram", "delhi ncr", "ncr",
}

# Work mode: role is hybrid — remote-only with no relocation willingness is bad
ACCEPTABLE_WORK_MODES: set[str] = {"hybrid", "onsite", "flexible"}
PENALISED_WORK_MODE   = "remote"   # Only penalised if also unwilling_to_relocate

LOCATION_SCORES: dict[str, float] = {
    "preferred":    1.00,   # Pune / Noida
    "acceptable_relocate": 0.80,   # Acceptable city + willing to relocate
    "acceptable_no_relocate": 0.65,  # Acceptable city, no relocation needed
    "other_relocate": 0.50,  # Other city but willing to relocate
    "remote_flexible": 0.35, # Remote but flexible/open
    "remote_locked": 0.10,   # Remote-only, refuses relocation, far city
}


# ─────────────────────────────────────────────────────────────────────────────
# JD: NOTICE PERIOD
# ─────────────────────────────────────────────────────────────────────────────
NOTICE_IDEAL_DAYS    = 30    # JD: "love sub-30 days, can buy out up to 30"
NOTICE_SOFT_MAX_DAYS = 60    # Still acceptable, minor penalty
NOTICE_HARD_MAX_DAYS = 90    # Significant penalty beyond this
# Score curve: 0-30d → 1.0, 31-60d → 0.8, 61-90d → 0.6, >90d → 0.35


# ─────────────────────────────────────────────────────────────────────────────
# JD: SKILLS — MUST-HAVE  (candidate needs ≥2 of these with real depth)
# ─────────────────────────────────────────────────────────────────────────────
MUST_HAVE_SKILLS: list[str] = [
    # Embeddings / retrieval
    "embeddings",
    "sentence transformers",
    "sentence-transformers",
    "information retrieval",
    "vector search",

    # Vector DBs / hybrid search
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "faiss",
    "elasticsearch",
    "opensearch",

    # Ranking / evaluation
    "ranking",
    "recommendation systems",
    "bm25",
    "ndcg",

    # Core language
    "python",
]

# Subset: these alone confirm retrieval/search production experience
RETRIEVAL_CORE_SKILLS: set[str] = {
    "embeddings", "sentence transformers", "sentence-transformers",
    "information retrieval", "vector search",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "elasticsearch", "opensearch", "bm25",
    "ranking", "recommendation systems",
}


# ─────────────────────────────────────────────────────────────────────────────
# JD: SKILLS — NICE-TO-HAVE  (bonus, not required)
# ─────────────────────────────────────────────────────────────────────────────
NICE_TO_HAVE_SKILLS: list[str] = [
    # LLM fine-tuning
    "fine-tuning llms",
    "lora",
    "qlora",
    "peft",
    "hugging face transformers",

    # Learning to rank
    "xgboost",
    "lightgbm",

    # Adjacent ML
    "nlp",
    "deep learning",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "mlops",
    "mlflow",
    "weights & biases",
    "langchain",
    "haystack",
    "prompt engineering",

    # Infra / scale
    "kafka",
    "spark",
    "kubernetes",
    "docker",
    "airflow",
]


# ─────────────────────────────────────────────────────────────────────────────
# JD: SKILLS — NEGATIVE SIGNALS (not disqualifiers alone, but drag score)
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_MISMATCH_SKILLS: list[str] = [
    # Computer vision / speech / robotics (JD explicitly calls this out)
    "computer vision",
    "image classification",
    "object detection",
    "cnn",
    "yolo",
    "opencv",
    "gans",
    "speech recognition",
    "tts",
    "reinforcement learning",
]


# ─────────────────────────────────────────────────────────────────────────────
# JD: CAREER — DISQUALIFYING COMPANIES (pure consulting only)
# A candidate is disqualified only if their ENTIRE career is at these firms.
# Having one stint here is fine if they also have product-company experience.
# ─────────────────────────────────────────────────────────────────────────────
CONSULTING_FIRMS: set[str] = {
    "tcs", "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant", "cognizant technology solutions",
    "capgemini",
    "hcl", "hcl technologies",
    "tech mahindra",
    "mphasis",
    "hexaware",
    "ltimindtree", "l&t infotech", "larsen & toubro infotech",
    "birlasoft",
    "cyient",
    "persistent systems",   # borderline — product mix, but often services
    "zensar",
    "mastech",
    "niit technologies",
}

# Industries that flag pure services / non-product roles
CONSULTING_INDUSTRIES: set[str] = {
    "it services",
    "consulting",
    "bpo",
    "outsourcing",
    "it outsourcing",
}

# Product-company indicator industries (boost career score)
PRODUCT_INDUSTRIES: set[str] = {
    "ai/ml",
    "software",
    "saas",
    "fintech",
    "e-commerce",
    "edtech",
    "healthtech",
    "hrtech",
    "marketplace",
    "internet",
    "product",
    "startup",
}


# ─────────────────────────────────────────────────────────────────────────────
# JD: CAREER — TITLE KEYWORDS (positive signals in title / description)
# ─────────────────────────────────────────────────────────────────────────────
POSITIVE_TITLE_KEYWORDS: list[str] = [
    "ai engineer",
    "ml engineer",
    "machine learning engineer",
    "applied scientist",
    "research engineer",
    "nlp engineer",
    "search engineer",
    "ranking engineer",
    "data scientist",
    "senior engineer",
    "staff engineer",
    "principal engineer",
    "founding engineer",
]

# Titles that are strong negative signals for THIS role
NEGATIVE_TITLE_KEYWORDS: list[str] = [
    "marketing",
    "sales",
    "operations",
    "accountant",
    "finance",
    "hr ",
    "recruiter",
    "customer support",
    "civil engineer",
    "mechanical engineer",
    "electrical engineer",
    "sap",
    "business analyst",    # not always — but rarely good fit
]

# Keywords in career descriptions that signal shipped retrieval/ranking systems
RETRIEVAL_SYSTEM_KEYWORDS: list[str] = [
    "search", "retrieval", "ranking", "recommendation",
    "embeddings", "vector", "semantic search",
    "elasticsearch", "opensearch", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus",
    "ndcg", "mrr", "a/b test", "relevance",
    "candidate matching", "job matching",
    "ranking system", "search system", "recommendation system",
    "bm25", "dense retrieval", "hybrid search",
    "re-ranking", "reranking",
]

# Production deployment keywords (JD cares about production, not just demos)
PRODUCTION_KEYWORDS: list[str] = [
    "production", "deployed", "scaled", "launched", "shipped",
    "serving", "inference", "real users", "live", "prod",
    "99th percentile", "latency", "throughput",
]


# ─────────────────────────────────────────────────────────────────────────────
# JD: CAREER — NEGATIVE CAREER PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
# Job-hopper: avg tenure < this → down-weight
MIN_AVG_TENURE_MONTHS = 18   # JD: "optimizing for titles by switching every 1.5 years"

# Title-chaser: >3 company switches in last 5 years is a signal
MAX_JOB_SWITCHES_5YRS = 3

# Production code gap: hasn't written code recently (JD: "this role writes code")
MAX_MANAGEMENT_GAP_MONTHS = 18  # If most recent role is pure management > this, penalty


# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIORAL SIGNALS — THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

# last_active_date: recency of platform activity
ACTIVE_DAYS_EXCELLENT  = 14   # active within 2 weeks  → score = 1.0
ACTIVE_DAYS_GOOD       = 30   # active within 1 month  → score = 0.85
ACTIVE_DAYS_ACCEPTABLE = 60   # active within 2 months → score = 0.65
ACTIVE_DAYS_STALE      = 90   # active within 3 months → score = 0.40
ACTIVE_DAYS_DEAD       = 180  # active within 6 months → score = 0.20
# Beyond 180 days → score = 0.05 (functionally unavailable)
SIGNAL_REFERENCE_DATE  = date(2026, 6, 17)   # today — used for recency calcs

# recruiter_response_rate: fraction 0.0–1.0
RESPONSE_RATE_EXCELLENT  = 0.70   # → 1.0
RESPONSE_RATE_GOOD       = 0.40   # → 0.75
RESPONSE_RATE_LOW        = 0.20   # → 0.40
RESPONSE_RATE_GHOST      = 0.10   # → 0.15 (effectively won't respond)

# avg_response_time_hours
RESPONSE_TIME_FAST_H     = 12    # ≤12h  → good signal
RESPONSE_TIME_OK_H       = 48    # ≤48h  → neutral
RESPONSE_TIME_SLOW_H     = 96    # ≤96h  → penalty
# >96h combined with low response_rate → ghost candidate

# interview_completion_rate
INTERVIEW_RATE_GOOD      = 0.80  # → no penalty
INTERVIEW_RATE_LOW       = 0.50  # → moderate penalty
INTERVIEW_RATE_BAD       = 0.30  # → strong penalty

# github_activity_score: -1 = no GitHub linked
GITHUB_WEIGHT            = 0.10  # weight inside signal sub-score
GITHUB_GOOD_THRESHOLD    = 50    # score ≥ 50 → good signal for this role
# -1 (no GitHub) for a founding AI Engineer role at a product company = mild negative

# profile_completeness_score
PROFILE_COMPLETE_MIN     = 60    # below this → mild penalty on signals


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

# Timeline sanity: career duration months vs stated years_of_experience
# Allow 20% slack (gaps, self-employment, etc.)
TIMELINE_SLACK_RATIO     = 1.40  # career_months > yoe_months * this → suspicious
TIMELINE_FLOOR_RATIO     = 0.50  # career_months < yoe_months * this → suspicious

# Skill duration sanity: "expert" skill with very little time used
EXPERT_MIN_DURATION_MONTHS   = 18   # expert with < 18 months → honeypot signal
ADVANCED_MIN_DURATION_MONTHS = 6    # advanced with < 6 months → honeypot signal

# Salary plausibility: INR LPA for a 5–9 yr AI Engineer in India
# Rough market range 2024–2025: 18–80 LPA
SALARY_MIN_PLAUSIBLE_LPA     = 5.0   # below this for claimed AI Engineer = suspicious
SALARY_MAX_PLAUSIBLE_LPA     = 150.0 # above this for any candidate = suspicious
SALARY_MIN_RANGE_RATIO       = 0.50  # min/max ratio: if max << min, data is corrupted

# Education year sanity: end_year must be after start_year
# Graduation year vs first job: can't have a job before you graduate (with 2yr slack)
EDUCATION_JOB_OVERLAP_SLACK_YRS = 1  # allow 1 year overlap (internships)

# max skills to be marked "expert" before it looks implausible
MAX_EXPERT_SKILLS        = 10   # >10 expert skills → mild honeypot signal

# Honeypot score threshold: candidates scoring above this are filtered out
HONEYPOT_SCORE_THRESHOLD = 3    # accumulate ≥3 honeypot flags → exclude


# ─────────────────────────────────────────────────────────────────────────────
# SKILL TRUST MODIFIERS
# Used to trust/distrust claimed skill proficiency
# ─────────────────────────────────────────────────────────────────────────────

# If platform assessment score exists, use it as a trust multiplier
# assessment_score (0–100) mapped to trust (0.0–1.0)
ASSESSMENT_TRUST_CURVE: list[tuple[int, float]] = [
    (0,   0.10),   # 0–19: very low trust
    (20,  0.30),
    (40,  0.55),
    (60,  0.75),
    (75,  0.90),
    (90,  1.00),   # 90+: full trust
]

# Endorsements: diminishing returns above 50
ENDORSEMENT_MAX_SCORE    = 50    # cap endorsements at this for scoring

# Proficiency level base weights (before trust modifiers)
PROFICIENCY_WEIGHTS: dict[str, float] = {
    "expert":       1.00,
    "advanced":     0.75,
    "intermediate": 0.50,
    "beginner":     0.20,
}


# ─────────────────────────────────────────────────────────────────────────────
# SUBMISSION OUTPUT FORMAT
# ─────────────────────────────────────────────────────────────────────────────
TOP_N              = 100    # number of candidates to output
CSV_COLUMNS        = ["candidate_id", "rank", "score", "reasoning"]
SCORE_DECIMAL_PLACES = 6   # round final scores to this many decimal places


# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME / PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE         = 5_000  # process candidates in batches for memory efficiency
MAX_RUNTIME_SECS   = 280    # 4m40s — leave 20s buffer under the 5-min hard limit


# ─────────────────────────────────────────────────────────────────────────────
# SCORING FLOOR / CAPS
# ─────────────────────────────────────────────────────────────────────────────
SCORE_MIN          = 0.0
SCORE_MAX          = 1.0
# Candidates flagged as honeypots are forced to 0.0
HONEYPOT_SCORE     = 0.0
# Disqualified (pure consulting, wrong domain) are capped at this
DISQUALIFIED_CAP   = 0.05
