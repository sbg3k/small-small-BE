import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, UUID4, RootModel
from supabase import create_client, Client
from google import genai

# --- CONFIGURATION ---

load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
SURVEY_FILE = os.getenv("SURVEY_FILE", "all.json")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("survey_api")

app = FastAPI(
    title="LLM-Assisted Mental Health Survey API",
    version="1.0.0",
    description="Administers PHQ-9, GAD-7, WHODAS, and analyzes freeform input with Gemini LLM.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- DEPENDENCIES ---

def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_gemini():
    return genai.Client()

# --- SCHEMAS ---

class Option(BaseModel):
    label: str
    value: Union[int, str]

class Question(BaseModel):
    id: str
    text: str
    type: Literal["radio", "textarea", "number", "interstitial"]
    options: Optional[List[Option]] = None
    optional: Optional[bool] = False
    min: Optional[int] = None
    max: Optional[int] = None

class Meta(BaseModel):
    title: str
    instructions: str

class SurveySection(BaseModel):
    meta: Meta
    questions: List[Question]

class SurveyFull(RootModel[List[SurveySection]]): pass

class PHQ9(BaseModel):
    phq1: int = Field(ge=0, le=3)
    phq2: int = Field(ge=0, le=3)
    phq3: int = Field(ge=0, le=3)
    phq4: int = Field(ge=0, le=3)
    phq5: int = Field(ge=0, le=3)
    phq6: int = Field(ge=0, le=3)
    phq7: int = Field(ge=0, le=3)
    phq8: int = Field(ge=0, le=3)
    phq9: int = Field(ge=0, le=3)

class GAD7(BaseModel):
    gad1: int = Field(ge=0, le=3)
    gad2: int = Field(ge=0, le=3)
    gad3: int = Field(ge=0, le=3)
    gad4: int = Field(ge=0, le=3)
    gad5: int = Field(ge=0, le=3)
    gad6: int = Field(ge=0, le=3)
    gad7: int = Field(ge=0, le=3)

class WHODAS(BaseModel):
    whodas1: int = Field(ge=0, le=4)
    whodas2: int = Field(ge=0, le=4)
    whodas3: int = Field(ge=0, le=4)
    whodas4: int = Field(ge=0, le=4)
    whodas5: int = Field(ge=0, le=4)
    whodas6: int = Field(ge=0, le=4)
    whodas7: int = Field(ge=0, le=4)
    whodas8: int = Field(ge=0, le=4)
    whodas9: int = Field(ge=0, le=4)
    whodas10: int = Field(ge=0, le=4)
    whodas11: int = Field(ge=0, le=4)
    whodas12: int = Field(ge=0, le=4)
    whodas_days_present: int = Field(ge=0, le=30)
    whodas_days_unable: int = Field(ge=0, le=30)
    whodas_days_cutback: int = Field(ge=0, le=30)

class Freeform(BaseModel):
    freeform_feelings: str
    freeform_challenges: str
    freeform_other_notes: str

class SurveySubmission(BaseModel):
    phq9: PHQ9
    gad7: GAD7
    whodas: WHODAS
    freeform: Freeform

class LLMSummary(BaseModel):
    depression: str
    anxiety: str
    functioning: str
    note: str

class SubmissionResponse(BaseModel):
    id: UUID4
    timestamp: datetime
    phq9_score: int
    gad7_score: int
    whodas_score: float
    llm_summary: LLMSummary

# --- BUSINESS LOGIC ---

def calculate_phq9_score(phq: PHQ9) -> int:
    return sum(phq.dict().values())

def calculate_gad7_score(gad: GAD7) -> int:
    return sum(gad.dict().values())

def calculate_whodas_score(whodas: WHODAS) -> float:
    return sum([v for k, v in whodas.dict().items() if k.startswith("whodas") and k[6].isdigit()])


def call_gemini(text: str, client) -> LLMSummary:
    prompt = f"""Classify the following mental health journal entry into structured categories.
    Entry:
    {text}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": LLMSummary,
            },
        )
        return response.parsed

    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini API error: {str(e)}"
        )

# --- ROUTES ---

@app.get("/health", summary="Health check", tags=["Utility"])
def health(
    supabase: Client = Depends(get_supabase),
    gemini_client = Depends(get_gemini),
):
    """Health check DB and LLM."""
    db_ok = False
    gemini_ok = False

    try:
        _ = supabase.table("survey_responses").select("*").limit(1).execute()
        db_ok = True
    except Exception as e:
        logger.warning(f"DB health check failed: {e}")

    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-pro",
            contents="ping"
        )
        gemini_ok = bool(resp.text.strip())
    except Exception as e:
        logger.warning(f"Gemini health check failed: {e}")

    return {
        "status": "ok" if db_ok and gemini_ok else "fail",
        "db": db_ok,
        "gemini": gemini_ok
    }

@app.get(
    "/survey",
    response_model=SurveyFull,
    response_model_exclude_unset=True,
    response_model_exclude_none=True,
    tags=["Survey"],
    summary="Get the full survey structure"
)
def get_full_questionnaire():
    try:
        with open(SURVEY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SurveyFull.model_validate(data)
    except FileNotFoundError:
        logger.error("Survey file not found.")
        raise HTTPException(status_code=500, detail="Survey file not found.")
    except json.JSONDecodeError:
        logger.error("Survey file is not valid JSON.")
        raise HTTPException(status_code=500, detail="Error parsing survey JSON.")
    except Exception as e:
        logger.error(f"Unknown error loading survey: {e}")
        raise HTTPException(status_code=500, detail=f"Survey data error: {str(e)}")

@app.post(
    "/survey/submit",
    response_model=SubmissionResponse,
    status_code=201,
    tags=["Survey"],
    summary="Submit survey responses and get scores/LLM output"
)
def submit_survey(
    data: SurveySubmission,
    supabase: Client = Depends(get_supabase),
    gemini_client = Depends(get_gemini)
):
    timestamp = datetime.utcnow()
    phq9_score = calculate_phq9_score(data.phq9)
    gad7_score = calculate_gad7_score(data.gad7)
    whodas_score = calculate_whodas_score(data.whodas)
    freeform_text = (
        f"{data.freeform.freeform_feelings}\n"
        f"{data.freeform.freeform_challenges}\n"
        f"{data.freeform.freeform_other_notes}"
    )
    llm_summary = call_gemini(freeform_text, gemini_client)
    db_payload = {
        "timestamp": timestamp.isoformat(),
        "phq9_score": phq9_score,
        "gad7_score": gad7_score,
        "whodas_score": whodas_score,
        "llm_depression": llm_summary.depression,
        "llm_anxiety": llm_summary.anxiety,
        "llm_functioning": llm_summary.functioning,
        "llm_note": llm_summary.note,
    }
    result = supabase.table("survey_responses").insert(db_payload).execute()
    if getattr(result, 'error', None):
        logger.error(f"Supabase insert error: {result.error}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to insert survey result: {result.error}"
        )
    inserted = result.data[0]
    return SubmissionResponse(
        id=inserted["id"],
        timestamp=timestamp,
        phq9_score=phq9_score,
        gad7_score=gad7_score,
        whodas_score=whodas_score,
        llm_summary=llm_summary
    )
