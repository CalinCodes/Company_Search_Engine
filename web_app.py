import os
import tempfile
import copy
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from stage1_filter import run as stage1_filter
from stage1_parser import parse_query
from stage2_retrieval import run as stage2_retrieval
from stage3_filter import run as stage3_filter


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "final_processed_data.json"

app = Flask(__name__)


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

    stage1_tmp = None
    stage2_tmp = None
    stage3_tmp = None
    try:
        parsed = parse_query(prompt)
        parsed["original_query"] = prompt

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

        results = []
        for index, company in enumerate(final, start=1):
            results.append(
                {
                    "rank": index,
                    "name": company.get("operational_name") or "N/A",
                    "company": company,
                }
            )

        return jsonify({
            "prompt": prompt,
            "total": len(results),
            "results": results,
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
    app.run(debug=True)
