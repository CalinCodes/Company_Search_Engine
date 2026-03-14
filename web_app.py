import os
import tempfile
import copy
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from stage1_filter import run as stage1_filter
from stage1_parser import (
    parse_query,
    should_skip_semantic_pipeline,
    get_explicit_prefilter_filters,
)
from stage2_retrieval import run as stage2_retrieval
from stage3_filter import run as stage3_filter
from translation import detect_language, translate, translate_results


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "final_processed_data.json"

app = Flask(__name__)


def _build_results(companies: list[dict]) -> list[dict]:
    results = []
    for index, company in enumerate(companies, start=1):
        results.append(
            {
                "rank": index,
                "name": company.get("operational_name") or "N/A",
                "company": company,
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


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/api/search")
def search():
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    if not os.environ.get("FEATHERLESS_API_KEY"):
        return jsonify({"error": "Missing FEATHERLESS_API_KEY in environment/.env."}), 500

    if not DATA_PATH.exists():
        return jsonify({"error": "final_processed_data.json file was not found."}), 500

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
            return jsonify(
                {
                    "prompt": prompt,
                    "detected_language": detected_lang,
                    "total": len(results),
                    "results": results,
                    "pipeline": "stage1_only",
                    "bypassed_stages": [2, 3],
                    "bypass_reason": parsed.get("execution_hints", {}).get("skip_reason"),
                }
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
            top_k=20,
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

        return jsonify({
            "prompt": prompt,
            "detected_language": detected_lang,
            "total": len(results),
            "results": results,
            "pipeline": "stage1_stage2_stage3",
            "prefilter_applied": prefilter_applied,
            "prefilter_filters": explicit_prefilter,
            "prefilter_candidate_count": len(filtered),
        })
    except Exception as exc:
        return jsonify({"error": f"Processing error: {str(exc)}"}), 500
    finally:
        for path in (stage1_tmp, stage2_tmp, stage3_tmp):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=25565, debug=True)
