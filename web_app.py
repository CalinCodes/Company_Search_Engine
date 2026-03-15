import os
import tempfile
import copy
import csv
import io
import time
import uuid
import threading
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory, send_file
from dotenv import load_dotenv
import pandas as pd
import stripe

from stage1_filter import run as stage1_filter
from stage1_parser import parse_query
from stage2_retrieval import run as stage2_retrieval
from stage3_filter import run as stage3_filter
from translation import detect_language, translate, translate_results

try:
    from stage1_parser import (
        should_skip_semantic_pipeline,
        get_explicit_prefilter_filters,
    )
except ImportError:
    # Backward compatibility with older stage1_parser versions.
    def should_skip_semantic_pipeline(parsed: dict) -> bool:
        hints = parsed.get("execution_hints", {}) if isinstance(parsed, dict) else {}
        return bool(hints.get("skip_semantic_pipeline"))

    def get_explicit_prefilter_filters(parsed: dict) -> dict:
        hints = parsed.get("execution_hints", {}) if isinstance(parsed, dict) else {}
        explicit = hints.get("explicit_prefilter_filters", {}) if isinstance(hints, dict) else {}
        if not isinstance(explicit, dict):
            return {}
        return {k: v for k, v in explicit.items() if v is not None}

try:
    import pycountry
except Exception:
    pycountry = None


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "final_processed_data.json"

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-secret-key-change-me")

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "").strip()
STRIPE_CURRENCY = os.environ.get("STRIPE_CURRENCY", "usd").strip().lower() or "usd"
DOWNLOAD_PRICE_CENTS = int(os.environ.get("DOWNLOAD_PRICE_CENTS", "500"))
PENDING_EXPORT_TTL_SECONDS = 30 * 60
CSV_SEP = ";"
HIDDEN_EXPORT_FIELDS = {"_filter_match", "_retrieval"}
PUBLIC_API_KEY = os.environ.get("PUBLIC_API_KEY", "").strip()
PUBLIC_API_ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.environ.get("PUBLIC_API_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
}

pending_exports: dict[str, dict] = {}
pending_exports_lock = threading.Lock()

JOB_TTL_SECONDS = 3600  # 1 hour
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def _cleanup_pending_exports() -> None:
    now = time.time()
    with pending_exports_lock:
        expired_tokens = [
            token
            for token, record in pending_exports.items()
            if now - float(record.get("created_at", now)) > PENDING_EXPORT_TTL_SECONDS
        ]
        for token in expired_tokens:
            pending_exports.pop(token, None)


def _cleanup_jobs() -> None:
    now = time.time()
    with jobs_lock:
        expired = [
            job_id
            for job_id, record in jobs.items()
            if now - float(record.get("created_at", now)) > JOB_TTL_SECONDS
        ]
        for job_id in expired:
            jobs.pop(job_id, None)


def _deliver_webhook(job_id: str, callback_url: str, payload: dict) -> None:
    try:
        requests.post(
            callback_url,
            json=payload,
            headers={"X-Job-Id": job_id, "Content-Type": "application/json"},
            timeout=10,
        )
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["callback_delivered"] = True
    except Exception:
        pass  # Best-effort delivery


def _run_pipeline_async(job_id: str, prompt: str, top_k: int, callback_url: str | None) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = "running"

    result, status_code = _run_search_pipeline(prompt=prompt, top_k=top_k)
    completed_at = time.time()

    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["completed_at"] = completed_at
            if status_code == 200:
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["result"] = result
            else:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = result.get("error", "Pipeline error")

    if callback_url:
        webhook_payload = {"job_id": job_id, "status": "completed" if status_code == 200 else "failed"}
        if status_code == 200:
            webhook_payload.update(result)
        else:
            webhook_payload["error"] = result.get("error", "Pipeline error")
        _deliver_webhook(job_id, callback_url, webhook_payload)


def _build_tabular_export_data(results: list[dict]) -> tuple[list[str], list[list]]:
    all_keys = set()
    for item in results:
        company = item.get("company") or {}
        if not isinstance(company, dict):
            continue
        for key in company.keys():
            if key not in HIDDEN_EXPORT_FIELDS:
                all_keys.add(key)

    ordered_company_keys = sorted(all_keys)
    headers = ["rank", "name", *ordered_company_keys]
    rows: list[list] = []

    for item in results:
        company = item.get("company") or {}
        if not isinstance(company, dict):
            company = {}

        row = [item.get("rank", ""), item.get("name", "")]
        for key in ordered_company_keys:
            value = company.get(key)
            if isinstance(value, (list, dict)):
                row.append(str(value))
            elif value is None:
                row.append("")
            else:
                row.append(value)
        rows.append(row)

    return headers, rows


def _csv_bytes(results: list[dict]) -> bytes:
    headers, rows = _build_tabular_export_data(results)
    stream = io.StringIO(newline="")
    stream.write(f"sep={CSV_SEP}\r\n")
    writer = csv.writer(stream, delimiter=CSV_SEP, quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode("utf-8-sig")


def _xlsx_bytes(results: list[dict]) -> bytes:
    headers, rows = _build_tabular_export_data(results)
    dataframe = pd.DataFrame(rows, columns=headers)
    stream = io.BytesIO()
    with pd.ExcelWriter(stream, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Companies")
    stream.seek(0)
    return stream.read()


def _country_name_from_code(code: str | None) -> str:
    """Convert ISO-2 code to full country name."""
    if not code:
        return ""

    code_norm = str(code).strip().upper()
    if len(code_norm) != 2:
        return str(code)

    if pycountry is None:
        return code_norm

    country = pycountry.countries.get(alpha_2=code_norm)
    return country.name if country else code_norm


def _enrich_company(company: dict) -> dict:
    """Attach frontend-friendly derived fields to company payload."""
    enriched = copy.deepcopy(company)
    code = enriched.get("address_country_code")
    enriched["address_country_name"] = _country_name_from_code(code)
    return enriched


def _build_results(companies: list[dict]) -> list[dict]:
    results = []
    for index, company in enumerate(companies, start=1):
        enriched_company = _enrich_company(company)
        results.append(
            {
                "rank": index,
                "name": enriched_company.get("operational_name") or "N/A",
                "company": enriched_company,
            }
        )
    return results


def _relaxed_query(parsed: dict) -> dict:
    relaxed = copy.deepcopy(parsed)
    filters = relaxed.setdefault("structured_filters", {})

    for key in [
        "business_models",
        "target_markets",
        "naics_codes",
        "min_employees",
        "max_employees",
        "min_revenue_usd",
        "max_revenue_usd",
    ]:
        filters[key] = None

    relaxed["reasoning"] = (
        f"{relaxed.get('reasoning', '').strip()} "
        "(Auto-relaxed filters after zero Stage 1 matches.)"
    ).strip()
    return relaxed


def _extract_public_api_key() -> str:
    api_key = (request.headers.get("X-API-Key") or "").strip()
    if api_key:
        return api_key

    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()

    return ""


def _is_public_api_authorized() -> bool:
    # If PUBLIC_API_KEY is not configured, public API stays open for local development.
    if not PUBLIC_API_KEY:
        return True
    return _extract_public_api_key() == PUBLIC_API_KEY


def _run_search_pipeline(prompt: str, top_k: int = 50) -> tuple[dict, int]:
    top_k = max(1, min(int(top_k), 100))

    if not prompt:
        return {"error": "Prompt is required."}, 400

    if not os.environ.get("FEATHERLESS_API_KEY"):
        return {"error": "Missing FEATHERLESS_API_KEY in environment/.env."}, 500

    if not DATA_PATH.exists():
        return {"error": "final_processed_data.json file was not found."}, 500

    # Detect language and translate query to English for the pipeline
    translate_key = os.environ.get("GOOGLE_TRANSLATE_API_KEY")
    detected_lang = "en"
    pipeline_prompt = prompt
    if translate_key:
        try:
            detected_lang = detect_language(prompt)
            if detected_lang != "en":
                pipeline_prompt = translate([prompt], target="en", source=detected_lang)[0]
        except Exception:
            pass  # Proceed with original prompt if detection fails

    stage1_tmp = None
    stage2_tmp = None
    stage3_tmp = None
    try:
        parsed = parse_query(pipeline_prompt)
        parsed["original_query"] = pipeline_prompt

        with tempfile.NamedTemporaryFile(suffix="_processed1.json", delete=False) as tmp1:
            stage1_tmp = tmp1.name

        with tempfile.NamedTemporaryFile(suffix="_processed2.json", delete=False) as tmp2:
            stage2_tmp = tmp2.name

        with tempfile.NamedTemporaryFile(suffix="_processed3.json", delete=False) as tmp3:
            stage3_tmp = tmp3.name

        filtered = stage1_filter(
            parsed,
            input_path=str(DATA_PATH),
            output_path=stage1_tmp,
        )

        if should_skip_semantic_pipeline(parsed):
            results = _build_results(filtered)
            if translate_key and detected_lang != "en":
                try:
                    results = translate_results(results, detected_lang)
                except Exception:
                    pass
            return (
                {
                    "prompt": prompt,
                    "detected_language": detected_lang,
                    "total": len(results),
                    "results": results,
                    "pipeline": "stage1_only",
                    "bypassed_stages": [2, 3],
                    "bypass_reason": parsed.get("execution_hints", {}).get("skip_reason"),
                },
                200,
            )

        explicit_prefilter = get_explicit_prefilter_filters(parsed)
        prefilter_applied = False
        if explicit_prefilter:
            prefilter_applied = True
            filtered = stage1_filter(
                {
                    "original_query": pipeline_prompt,
                    "structured_filters": explicit_prefilter,
                    "semantic_keywords": parsed.get("semantic_keywords", []),
                    "role_label": parsed.get("role_label", "Unknown"),
                    "reasoning": parsed.get("reasoning", ""),
                },
                input_path=str(DATA_PATH),
                output_path=stage1_tmp,
            )

        if not filtered:
            parsed = _relaxed_query(parsed)
            filtered = stage1_filter(
                parsed,
                input_path=str(DATA_PATH),
                output_path=stage1_tmp,
            )

        ranked = stage2_retrieval(
            parsed,
            input_path=stage1_tmp,
            output_path=stage2_tmp,
            top_k=top_k,
        )

        final = stage3_filter(
            parsed,
            input_path=stage2_tmp,
            output_path=stage3_tmp,
        )

        results = _build_results(final)
        if translate_key and detected_lang != "en":
            try:
                results = translate_results(results, detected_lang)
            except Exception:
                pass

        return (
            {
                "prompt": prompt,
                "detected_language": detected_lang,
                "total": len(results),
                "results": results,
                "pipeline": "stage1_stage2_stage3",
                "prefilter_applied": prefilter_applied,
                "prefilter_filters": explicit_prefilter,
                "prefilter_candidate_count": len(filtered),
            },
            200,
        )
    except Exception as exc:
        return {"error": f"Processing error: {str(exc)}"}, 500
    finally:
        for path in (stage1_tmp, stage2_tmp, stage3_tmp):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


@app.after_request
def add_public_api_cors_headers(response):
    if not request.path.startswith("/api/public/"):
        return response

    origin = (request.headers.get("Origin") or "").strip()
    if "*" in PUBLIC_API_ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = "*"
    elif origin and origin in PUBLIC_API_ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"

    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-API-Key"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/api/search")
def search():
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    payload, status = _run_search_pipeline(prompt=prompt, top_k=50)
    return jsonify(payload), status


@app.route("/api/public/health", methods=["GET", "OPTIONS"])
def public_health():
    if request.method == "OPTIONS":
        return ("", 204)

    return jsonify({
        "status": "ok",
        "service": "veridion-public-api",
        "api_key_required": False,
    })


@app.route("/api/public/search", methods=["POST", "OPTIONS"])
def public_search():
    if request.method == "OPTIONS":
        return ("", 204)

    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    top_k = body.get("top_k", 50)
    callback_url = (body.get("callback_url") or "").strip() or None

    try:
        top_k_value = int(top_k)
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer."}), 400

    if callback_url:
        if not callback_url.startswith(("http://", "https://")):
            return jsonify({"error": "callback_url must be an http or https URL."}), 400

        _cleanup_jobs()
        job_id = uuid.uuid4().hex
        with jobs_lock:
            jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "created_at": time.time(),
                "completed_at": None,
                "prompt": prompt,
                "top_k": top_k_value,
                "callback_url": callback_url,
                "result": None,
                "error": None,
                "callback_delivered": False,
            }

        thread = threading.Thread(
            target=_run_pipeline_async,
            args=(job_id, prompt, top_k_value, callback_url),
            daemon=True,
        )
        thread.start()

        return jsonify({
            "job_id": job_id,
            "status": "pending",
            "poll_url": f"/api/public/jobs/{job_id}",
        }), 202

    payload, status = _run_search_pipeline(prompt=prompt, top_k=top_k_value)
    return jsonify(payload), status


@app.route("/api/public/jobs/<job_id>", methods=["GET", "OPTIONS"])
def get_job(job_id: str):
    if request.method == "OPTIONS":
        return ("", 204)

    _cleanup_jobs()
    with jobs_lock:
        record = jobs.get(job_id)

    if not record:
        return jsonify({"error": "Job not found or expired."}), 404

    response: dict = {
        "job_id": job_id,
        "status": record["status"],
        "prompt": record["prompt"],
        "created_at": record["created_at"],
        "completed_at": record["completed_at"],
        "callback_delivered": record["callback_delivered"],
    }

    if record["status"] == "completed":
        response.update(record["result"] or {})
    elif record["status"] == "failed":
        response["error"] = record["error"]

    return jsonify(response), 200


@app.post("/api/create-checkout-session")
def create_checkout_session():
    if not stripe.api_key:
        return jsonify({"error": "Missing STRIPE_SECRET_KEY in environment/.env."}), 500

    body = request.get_json(silent=True) or {}
    export_format = str(body.get("format") or "").strip().lower()
    results = body.get("results")

    if export_format not in {"csv", "xlsx"}:
        return jsonify({"error": "Invalid format. Use csv or xlsx."}), 400

    if not isinstance(results, list) or not results:
        return jsonify({"error": "No results available for export."}), 400

    _cleanup_pending_exports()

    token = uuid.uuid4().hex
    pending_record = {
        "format": export_format,
        "results": results,
        "created_at": time.time(),
        "is_paid": False,
        "is_downloaded": False,
        "stripe_session_id": None,
    }

    base_url = request.host_url.rstrip("/")
    success_url = (
        f"{base_url}/?payment=success&token={token}&session_id={{CHECKOUT_SESSION_ID}}"
    )
    cancel_url = f"{base_url}/?payment=cancel"

    line_items = []
    if STRIPE_PRICE_ID:
        line_items = [{"price": STRIPE_PRICE_ID, "quantity": 1}]
    else:
        line_items = [
            {
                "price_data": {
                    "currency": STRIPE_CURRENCY,
                    "product_data": {"name": f"{export_format.upper()} export"},
                    "unit_amount": DOWNLOAD_PRICE_CENTS,
                },
                "quantity": 1,
            }
        ]

    try:
        checkout_session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            line_items=line_items,
            success_url=success_url,
            cancel_url=cancel_url,
        )
    except Exception as exc:
        return jsonify({"error": f"Stripe checkout creation failed: {str(exc)}"}), 500

    pending_record["stripe_session_id"] = checkout_session.get("id")
    with pending_exports_lock:
        pending_exports[token] = pending_record

    return jsonify({"checkout_url": checkout_session.get("url")})


@app.post("/api/confirm-payment")
def confirm_payment():
    if not stripe.api_key:
        return jsonify({"error": "Missing STRIPE_SECRET_KEY in environment/.env."}), 500

    body = request.get_json(silent=True) or {}
    token = str(body.get("token") or "").strip()
    session_id = str(body.get("session_id") or "").strip()

    if not token or not session_id:
        return jsonify({"error": "Missing token or session_id."}), 400

    _cleanup_pending_exports()

    with pending_exports_lock:
        pending_record = pending_exports.get(token)

    if not pending_record:
        return jsonify({"error": "Export session expired. Please start checkout again."}), 404

    expected_session_id = pending_record.get("stripe_session_id")
    if expected_session_id != session_id:
        return jsonify({"error": "Session mismatch."}), 400

    try:
        checkout_session = stripe.checkout.Session.retrieve(session_id)
    except Exception as exc:
        return jsonify({"error": f"Stripe session verification failed: {str(exc)}"}), 500

    if checkout_session.get("payment_status") != "paid":
        return jsonify({"error": "Payment not completed."}), 402

    with pending_exports_lock:
        if token in pending_exports:
            pending_exports[token]["is_paid"] = True

    return jsonify({"download_url": f"/api/download-paid?token={token}"})


@app.get("/api/download-paid")
def download_paid_export():
    token = (request.args.get("token") or "").strip()
    if not token:
        return jsonify({"error": "Missing token."}), 400

    _cleanup_pending_exports()

    with pending_exports_lock:
        pending_record = pending_exports.get(token)

    if not pending_record:
        return jsonify({"error": "Export session expired. Please start checkout again."}), 404

    if not pending_record.get("is_paid"):
        return jsonify({"error": "Payment required before download."}), 402

    if pending_record.get("is_downloaded"):
        return jsonify({"error": "This one-time download has already been used."}), 410

    export_format = pending_record.get("format")
    export_results = pending_record.get("results") or []
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    try:
        if export_format == "csv":
            payload = _csv_bytes(export_results)
            mimetype = "text/csv"
            filename = f"companies-{timestamp}.csv"
        else:
            payload = _xlsx_bytes(export_results)
            mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"companies-{timestamp}.xlsx"
    except Exception as exc:
        return jsonify({"error": f"Failed to prepare export file: {str(exc)}"}), 500

    with pending_exports_lock:
        if token in pending_exports:
            pending_exports[token]["is_downloaded"] = True

    return send_file(
        io.BytesIO(payload),
        as_attachment=True,
        download_name=filename,
        mimetype=mimetype,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=25565, debug=True)
