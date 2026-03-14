import os
import json
import ast
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from typing import Optional

FEATHERLESS_API_KEY = os.environ.get("FEATHERLESS_API_KEY")
FEATHERLESS_MODEL = "Qwen/Qwen3-32B"
MAX_WORKERS = 2

client = None
if FEATHERLESS_API_KEY:
    client = OpenAI(
        base_url="https://api.featherless.ai/v1",
        api_key=FEATHERLESS_API_KEY,
    )

# --- API ENRICHMENT LAYER ---
_llm_error_logged = False

def enrich_from_llm(company_name, website, description, missing_fields):
    """Use Qwen3 via Featherless AI to estimate missing company fields."""
    global _llm_error_logged
    if not client:
        return {}

    fields_desc = ", ".join(missing_fields)
    # /no_think disables Qwen3 chain-of-thought to save tokens and avoid <think> blocks
    prompt = (
        f"/no_think\n"
        f"You are a company data research assistant. Given the following company info, "
        f"estimate the missing fields: {fields_desc}.\n\n"
        f"Company: {company_name}\n"
        f"Website: {website}\n"
        f"Description: {description}\n\n"
        f"Respond ONLY with a JSON object. Use null for any field you cannot estimate.\n"
        f"Available keys: revenue (annual USD as a number), employee_count (integer), "
        f"operational_name (string), year_founded (integer)\n"
        f'Example: {{"revenue": 5000000, "employee_count": 150, "operational_name": "Acme Corp", "year_founded": 2005}}\n'
        f"JSON only, no explanation, no markdown."
    )

    try:
        response = client.chat.completions.create(
            model=FEATHERLESS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        # Strip <think>...</think> blocks if the model still emits them
        if "<think>" in text:
            text = text[text.rfind("</think>") + len("</think>"):].strip()
        # Strip markdown code blocks if present
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except Exception as e:
        if not _llm_error_logged:
            tqdm.write(f"\n[LLM error] {type(e).__name__}: {e}")
            _llm_error_logged = True
        return {}

# --- VALIDATION LAYER ---
class CompanySchema(BaseModel):
    """The 'Golden Standard' for your data."""
    operational_name: str
    website: str
    address_country_code: str = Field(min_length=2, max_length=2)
    revenue: float = Field(ge=0)
    employee_count: Optional[int] = Field(None, ge=0)
    primary_naics_code: str

    @field_validator('website')
    @classmethod
    def validate_domain(cls, v: str):
        if "." not in v: raise ValueError("Invalid domain")
        return v

# --- PARSING LOGIC ---
def parse_dict_field(value):
    if isinstance(value, dict): return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value.strip())
            return parsed if isinstance(parsed, dict) else {}
        except: return {}
    return {}

def process_row(idx, row):
    """Process and validate a single company row. Returns (record, error, was_enriched)."""
    addr = parse_dict_field(row.get('address', {}))
    naics = parse_dict_field(row.get('primary_naics', {}))

    website = row.get('website')
    name = row.get('operational_name')
    revenue = row.get('revenue')
    employee_count = row.get('employee_count')

    # Identify missing fields the LLM can help fill
    missing_fields = []
    if pd.isna(revenue):
        missing_fields.append("revenue")
    if pd.isna(employee_count):
        missing_fields.append("employee_count")
    if pd.isna(name):
        missing_fields.append("operational_name")

    # Enrich nulls via LLM
    was_enriched = False
    if FEATHERLESS_API_KEY and missing_fields:
        comp_name = name if pd.notna(name) else "Unknown"
        comp_website = website if pd.notna(website) else "Unknown"
        comp_desc = row.get('description') or ''
        enriched = enrich_from_llm(comp_name, comp_website, comp_desc, missing_fields)
        if enriched:
            filled = False
            if pd.isna(revenue) and enriched.get('revenue') is not None:
                revenue = enriched['revenue']
                filled = True
            if pd.isna(employee_count) and enriched.get('employee_count') is not None:
                employee_count = enriched['employee_count']
                filled = True
            if pd.isna(name) and enriched.get('operational_name'):
                name = enriched['operational_name']
                filled = True
            was_enriched = filled

    company_dict = {
        "_orig_idx": idx,
        "website": website,
        "operational_name": name,
        "address_country_code": addr.get('country_code'),
        "address_latitude": addr.get('latitude'),
        "address_longitude": addr.get('longitude'),
        "revenue": revenue if pd.notna(revenue) else None,
        "employee_count": int(employee_count) if pd.notna(employee_count) else None,
        "primary_naics_code": naics.get('code'),
        "description": row.get('description'),
    }

    error = None
    try:
        CompanySchema(**{k: v for k, v in company_dict.items() if k != '_orig_idx'})
    except Exception as e:
        error = {"company": name or 'Unknown', "error": str(e)}

    # Always return the enriched record regardless of validation result
    return company_dict, error, was_enriched


def process_and_validate(json_file):
    df = pd.read_json(json_file)
    rows = [(idx, row) for idx, row in df.iterrows()]

    if not FEATHERLESS_API_KEY:
        print("Warning: FEATHERLESS_API_KEY not set. Skipping API enrichment.")

    all_records = []
    errors = []
    enriched_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, idx, row): idx for idx, row in rows}
        with tqdm(total=len(rows), desc="Processing companies", unit="co") as pbar:
            for future in as_completed(futures):
                record, error, was_enriched = future.result()
                all_records.append(record)
                if error:
                    errors.append(error)
                if was_enriched:
                    enriched_count += 1
                valid_count = len(all_records) - len(errors)
                pbar.set_postfix(ok=valid_count, flagged=len(errors), enriched=enriched_count)
                pbar.update(1)

    # Sort by original index to preserve row order for index-based joining
    all_records.sort(key=lambda r: r['_orig_idx'])
    pd.DataFrame(all_records).to_json('processed_data.json', orient='records', indent=2)

    print(f"\nEnriched {enriched_count} companies via Featherless AI ({FEATHERLESS_MODEL}).")
    print(f"Successfully validated {len(all_records) - len(errors)} companies.")
    print(f"Flagged {len(errors)} companies for review.")
    return errors

# Run it
error_log = process_and_validate('data.json')

# for error in error_log:
#     print(f"Company: {error['company']} | Issue: {error['error']}")
