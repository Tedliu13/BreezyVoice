# OpenAI API Spec. Reference: https://platform.openai.com/docs/api-reference/audio/createSpeech

import csv
import json
import os
import re
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import onnxruntime
import torchaudio
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from g2pw import G2PWConverter
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from cosyvoice.utils.file_utils import load_wav
from single_inference import CustomCosyVoice, get_bopomofo_rare


class Settings(BaseSettings):
    api_key: str = Field(
        default="", description="Specifies the API key used to authenticate the user."
    )

    model_path: str = Field(
        default="MediaTek-Research/BreezyVoice",
        description="Specifies the model used for speech synthesis.",
    )
    speaker_prompt_audio_path: str = Field(
        default="./data/example.wav",
        description="Specifies the path to the prompt speech audio file of the speaker.",
    )
    speaker_prompt_text_transcription: str = Field(
        default="在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。只有擁有解密方法的對象，經由解密過程，才能將密文還原為正常可讀的內容。",
        description="Specifies the transcription of the speaker prompt audio.",
    )
    preload_model: bool = Field(
        default=False,
        description="Whether to preload the TTS model during application startup.",
    )
    idle_unload_seconds: int = Field(
        default=600,
        description="Unload the TTS runtime after this many idle seconds. Set 0 to disable.",
    )
    stats_file_path: str = Field(
        default="/root/.cache/huggingface/breezyvoice-stats.json",
        description="Path to the persistent stats file. Put this on a mounted volume if you want counters to survive restarts.",
    )


class SpeechRequest(BaseModel):
    model: str = ""
    input: str = Field(
        description="The content that will be synthesized into speech. You can include phonetic symbols if needed, though they should be used sparingly.",
        examples=["今天天氣真好"],
    )
    response_format: str = ""
    speed: float = 1.0


class BatchJobResponse(BaseModel):
    job_id: str
    status: str
    completed: int
    total: int
    current: int = 0
    current_filename: str | None = None
    download_url: str | None = None
    error: str | None = None


class RuntimeStatusResponse(BaseModel):
    model_status: str
    active_users: int
    usage_count: int
    converted_files_count: int


def require_api_key(request: Request) -> None:
    expected_api_key = request.app.state.settings.api_key
    if not expected_api_key:
        return

    auth_header = request.headers.get("authorization", "")
    scheme, _, provided_api_key = auth_header.partition(" ")
    if scheme.lower() != "bearer" or provided_api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def normalize_text(cosyvoice: CustomCosyVoice, text: str) -> str:
    return cosyvoice.frontend.text_normalize_new(text.strip(), split=False)


def to_bopomofo(cosyvoice: CustomCosyVoice, bopomofo_converter: G2PWConverter, text: str) -> str:
    normalized_text = normalize_text(cosyvoice, text)
    return get_bopomofo_rare(normalized_text, bopomofo_converter)


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "output"


def read_prompt_audio(upload: UploadFile) -> object:
    suffix = Path(upload.filename or "prompt.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(upload.file.read())
        temp_path = temp_file.name
    try:
        return load_wav(temp_path, 16000)
    finally:
        os.unlink(temp_path)


def synthesize_wav_bytes(
    cosyvoice: CustomCosyVoice,
    prompt_speech_16k,
    prompt_text_bopomo: str,
    content_to_synthesize: str,
    bopomofo_converter: G2PWConverter,
) -> bytes:
    content_to_synthesize_bopomo = to_bopomofo(
        cosyvoice, bopomofo_converter, content_to_synthesize
    )
    output = cosyvoice.inference_zero_shot_no_normalize(
        content_to_synthesize_bopomo,
        prompt_text_bopomo,
        prompt_speech_16k,
    )
    audio_buffer = BytesIO()
    torchaudio.save(audio_buffer, output["tts_speech"], 22050, format="wav")
    return audio_buffer.getvalue()


def parse_batch_rows(csv_bytes: bytes) -> list[dict[str, str]]:
    try:
        decoded = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV must be UTF-8 encoded.",
        ) from exc

    reader = csv.DictReader(BytesIO(decoded.encode("utf-8")).read().decode("utf-8").splitlines())
    if not reader.fieldnames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV must include a header row.",
        )

    field_lookup = {field.strip().lower(): field for field in reader.fieldnames if field}
    text_field = None
    for candidate in ("text", "content_to_synthesize", "content", "input"):
        if candidate in field_lookup:
            text_field = field_lookup[candidate]
            break
    if text_field is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV must include one of these columns: text, content_to_synthesize, content, input.",
        )

    filename_field = None
    for candidate in ("filename", "output_audio_filename", "output_filename", "id", "name"):
        if candidate in field_lookup:
            filename_field = field_lookup[candidate]
            break

    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader, start=1):
        text = (row.get(text_field) or "").strip()
        if not text:
            continue
        if filename_field:
            filename = sanitize_filename(row.get(filename_field) or f"row-{index:03d}")
        else:
            filename = f"row-{index:03d}"
        rows.append({"filename": filename, "text": text})

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV does not contain any non-empty text rows.",
        )
    return rows


def create_bopomofo_converter() -> G2PWConverter:
    original_inference_session = onnxruntime.InferenceSession
    available_providers = onnxruntime.get_available_providers()
    default_providers = ["CPUExecutionProvider"]
    if "CPUExecutionProvider" not in available_providers and available_providers:
        default_providers = [available_providers[0]]

    def patched_inference_session(*args: Any, **kwargs: Any):
        kwargs.setdefault("providers", default_providers)
        return original_inference_session(*args, **kwargs)

    onnxruntime.InferenceSession = patched_inference_session
    try:
        return G2PWConverter()
    finally:
        onnxruntime.InferenceSession = original_inference_session


def set_runtime_status(app: FastAPI, status_text: str) -> None:
    app.state.runtime_status = status_text


def load_persistent_stats(app: FastAPI) -> None:
    stats_path = Path(app.state.settings.stats_file_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if not stats_path.exists():
        app.state.stats_file_path = stats_path
        app.state.usage_count = 0
        app.state.converted_files_count = 0
        return

    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

    app.state.stats_file_path = stats_path
    app.state.usage_count = int(payload.get("usage_count", 0) or 0)
    app.state.converted_files_count = int(payload.get("converted_files_count", 0) or 0)


def save_persistent_stats(app: FastAPI) -> None:
    stats_path = app.state.stats_file_path
    payload = {
        "usage_count": app.state.usage_count,
        "converted_files_count": app.state.converted_files_count,
    }
    stats_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_runtime(app: FastAPI) -> None:
    if getattr(app.state, "runtime_ready", False):
        app.state.runtime_last_used = time.time()
        return

    with app.state.runtime_lock:
        if getattr(app.state, "runtime_ready", False):
            app.state.runtime_last_used = time.time()
            return
        set_runtime_status(app, "載入中")
        app.state.runtime_loading = True
        try:
            app.state.cosyvoice = CustomCosyVoice(app.state.settings.model_path)
            app.state.bopomofo_converter = create_bopomofo_converter()
            app.state.prompt_speech_16k = load_wav(
                app.state.settings.speaker_prompt_audio_path, 16000
            )
            app.state.runtime_ready = True
            app.state.runtime_last_used = time.time()
            if app.state.active_users > 0:
                set_runtime_status(app, "使用中")
            else:
                set_runtime_status(app, "已載入")
        except Exception:
            app.state.runtime_ready = False
            app.state.runtime_last_used = 0.0
            set_runtime_status(app, "未載入")
            raise
        finally:
            app.state.runtime_loading = False


def touch_runtime(app: FastAPI) -> None:
    app.state.runtime_last_used = time.time()


def unload_runtime(app: FastAPI) -> None:
    with app.state.runtime_lock:
        if not getattr(app.state, "runtime_ready", False):
            return
        if getattr(app.state, "active_users", 0) > 0:
            return
        del app.state.cosyvoice
        del app.state.bopomofo_converter
        del app.state.prompt_speech_16k
        app.state.runtime_ready = False
        app.state.runtime_last_used = 0.0
        app.state.runtime_loading = False
        set_runtime_status(app, "已閒置卸載")


def begin_usage(app: FastAPI) -> None:
    with app.state.metrics_lock:
        app.state.active_users += 1
    app.state.runtime_last_used = time.time()
    if getattr(app.state, "runtime_ready", False):
        set_runtime_status(app, "使用中")


def end_usage(app: FastAPI) -> None:
    with app.state.metrics_lock:
        app.state.active_users = max(0, app.state.active_users - 1)
        active_users = app.state.active_users
    app.state.runtime_last_used = time.time()
    if getattr(app.state, "runtime_ready", False):
        if active_users > 0:
            set_runtime_status(app, "使用中")
        else:
            set_runtime_status(app, "已載入")


def record_usage(app: FastAPI, converted_files: int = 0) -> None:
    with app.state.metrics_lock:
        app.state.usage_count += 1
        app.state.converted_files_count += converted_files
        save_persistent_stats(app)


def record_converted_files(app: FastAPI, converted_files: int) -> None:
    with app.state.metrics_lock:
        app.state.converted_files_count += converted_files
        save_persistent_stats(app)


def get_runtime_status_payload(app: FastAPI) -> RuntimeStatusResponse:
    with app.state.metrics_lock:
        active_users = app.state.active_users
        usage_count = app.state.usage_count
        converted_files_count = app.state.converted_files_count
    return RuntimeStatusResponse(
        model_status=app.state.runtime_status,
        active_users=active_users,
        usage_count=usage_count,
        converted_files_count=converted_files_count,
    )


def runtime_idle_monitor(app: FastAPI) -> None:
    while not getattr(app.state, "shutdown_requested", False):
        idle_unload_seconds = app.state.settings.idle_unload_seconds
        if idle_unload_seconds > 0 and getattr(app.state, "runtime_ready", False):
            if getattr(app.state, "active_users", 0) > 0:
                time.sleep(5)
                continue
            last_used = getattr(app.state, "runtime_last_used", 0.0)
            if last_used and (time.time() - last_used) >= idle_unload_seconds:
                unload_runtime(app)
        time.sleep(5)


JOB_STORE: dict[str, dict[str, Any]] = {}
JOB_STORE_LOCK = threading.Lock()
JOB_ROOT = Path(tempfile.gettempdir()) / "breezyvoice-batch-jobs"
JOB_ROOT.mkdir(parents=True, exist_ok=True)


def get_job(job_id: str) -> dict[str, Any]:
    with JOB_STORE_LOCK:
        job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch job not found.",
        )
    return job


def serialize_job(job_id: str, job: dict[str, Any]) -> BatchJobResponse:
    download_url = None
    if job["status"] == "completed":
        download_url = f"/v1/batch/jobs/{job_id}/download"
    return BatchJobResponse(
        job_id=job_id,
        status=job["status"],
        completed=job["completed"],
        total=job["total"],
        current=job.get("current", 0),
        current_filename=job.get("current_filename"),
        download_url=download_url,
        error=job.get("error"),
    )


def set_job_state(job_id: str, **updates: Any) -> dict[str, Any]:
    with JOB_STORE_LOCK:
        job = JOB_STORE[job_id]
        job.update(updates)
        return dict(job)


def create_batch_job(
    csv_bytes: bytes,
    speaker_prompt_audio_bytes: bytes,
    speaker_prompt_audio_filename: str,
    speaker_prompt_text_transcription: str,
) -> str:
    rows = parse_batch_rows(csv_bytes)
    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    prompt_suffix = Path(speaker_prompt_audio_filename or "prompt.wav").suffix or ".wav"
    prompt_path = job_dir / f"prompt{prompt_suffix}"
    prompt_path.write_bytes(speaker_prompt_audio_bytes)
    csv_path = job_dir / "input.csv"
    csv_path.write_bytes(csv_bytes)

    with JOB_STORE_LOCK:
        JOB_STORE[job_id] = {
            "status": "queued",
            "completed": 0,
            "total": len(rows),
            "current": 0,
            "current_filename": None,
            "rows": rows,
            "prompt_path": str(prompt_path),
            "prompt_text": speaker_prompt_text_transcription,
            "zip_path": str(job_dir / "breezyvoice-batch.zip"),
            "error": None,
        }
    return job_id


def run_batch_job(app: FastAPI, job_id: str) -> None:
    job = get_job(job_id)
    set_job_state(job_id, status="processing")
    try:
        begin_usage(app)
        ensure_runtime(app)
        prompt_speech_16k = load_wav(job["prompt_path"], 16000)
        prompt_text_bopomo = to_bopomofo(
            app.state.cosyvoice,
            app.state.bopomofo_converter,
            job["prompt_text"],
        )

        zip_path = Path(job["zip_path"])
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            manifest_rows = ["filename,text"]
            for index, row in enumerate(job["rows"], start=1):
                set_job_state(
                    job_id,
                    current=index,
                    current_filename=f"{row['filename']}.wav",
                )
                wav_bytes = synthesize_wav_bytes(
                    app.state.cosyvoice,
                    prompt_speech_16k,
                    prompt_text_bopomo,
                    row["text"],
                    app.state.bopomofo_converter,
                )
                zip_file.writestr(f"{row['filename']}.wav", wav_bytes)
                escaped_text = row["text"].replace('"', '""')
                manifest_rows.append(f'{row["filename"]}.wav,"{escaped_text}"')
                set_job_state(job_id, completed=index)
                touch_runtime(app)
                record_converted_files(app, 1)
            zip_file.writestr("manifest.csv", "\n".join(manifest_rows).encode("utf-8"))

        set_job_state(job_id, status="completed", current=job["total"])
    except Exception as exc:
        set_job_state(job_id, status="failed", error=str(exc))
    finally:
        end_usage(app)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    app.state.runtime_lock = threading.Lock()
    app.state.metrics_lock = threading.Lock()
    app.state.runtime_ready = False
    app.state.runtime_loading = False
    app.state.runtime_last_used = 0.0
    app.state.runtime_status = "未載入"
    app.state.active_users = 0
    load_persistent_stats(app)
    app.state.shutdown_requested = False
    app.state.runtime_monitor = threading.Thread(
        target=runtime_idle_monitor,
        args=(app,),
        daemon=True,
    )
    app.state.runtime_monitor.start()
    if app.state.settings.preload_model:
        ensure_runtime(app)
    yield
    app.state.shutdown_requested = True
    unload_runtime(app)


app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/v1")
WEB_DIR = Path(__file__).resolve().parent / "web"


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(WEB_DIR / "index.html")


@router.get("/runtime/status", response_model=RuntimeStatusResponse)
async def runtime_status_endpoint():
    return get_runtime_status_payload(app)


@router.get("/models")
async def get_models(request: Request):
    require_api_key(request)
    return {
        "object": "list",
        "data": [
            {
                "id": request.app.state.settings.model_path,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@router.post("/audio/speech")
async def speach_endpoint(request: Request, payload: SpeechRequest):
    require_api_key(request)
    begin_usage(request.app)
    ensure_runtime(request.app)
    try:
        prompt_text_bopomo = to_bopomofo(
            request.app.state.cosyvoice,
            request.app.state.bopomofo_converter,
            request.app.state.settings.speaker_prompt_text_transcription,
        )
        audio_bytes = synthesize_wav_bytes(
            request.app.state.cosyvoice,
            request.app.state.prompt_speech_16k,
            prompt_text_bopomo,
            payload.input,
            request.app.state.bopomofo_converter,
        )
        touch_runtime(request.app)
        record_usage(request.app, converted_files=1)
        return StreamingResponse(
            BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )
    finally:
        end_usage(request.app)


@router.post("/batch/speech")
async def batch_speech_endpoint(
    request: Request,
    speaker_prompt_audio: UploadFile = File(...),
    speaker_prompt_text_transcription: str = Form(...),
    batch_csv: UploadFile = File(...),
):
    require_api_key(request)
    begin_usage(request.app)
    try:
        ensure_runtime(request.app)

        prompt_text = speaker_prompt_text_transcription.strip()
        if not prompt_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="speaker_prompt_text_transcription is required.",
            )

        csv_bytes = await batch_csv.read()
        rows = parse_batch_rows(csv_bytes)

        prompt_speech_16k = read_prompt_audio(speaker_prompt_audio)
        prompt_text_bopomo = to_bopomofo(
            request.app.state.cosyvoice,
            request.app.state.bopomofo_converter,
            prompt_text,
        )

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            manifest_rows = ["filename,text"]
            for row in rows:
                wav_bytes = synthesize_wav_bytes(
                    request.app.state.cosyvoice,
                    prompt_speech_16k,
                    prompt_text_bopomo,
                    row["text"],
                    request.app.state.bopomofo_converter,
                )
                zip_file.writestr(f"{row['filename']}.wav", wav_bytes)
                escaped_text = row["text"].replace('"', '""')
                manifest_rows.append(f'{row["filename"]}.wav,"{escaped_text}"')
                touch_runtime(request.app)
            zip_file.writestr("manifest.csv", "\n".join(manifest_rows).encode("utf-8"))

        zip_buffer.seek(0)
        record_usage(request.app, converted_files=len(rows))
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=breezyvoice-batch.zip"},
        )
    finally:
        end_usage(request.app)


@router.post("/batch/jobs", response_model=BatchJobResponse)
async def create_batch_job_endpoint(
    request: Request,
    speaker_prompt_audio: UploadFile = File(...),
    speaker_prompt_text_transcription: str = Form(...),
    batch_csv: UploadFile = File(...),
):
    require_api_key(request)

    prompt_text = speaker_prompt_text_transcription.strip()
    if not prompt_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="speaker_prompt_text_transcription is required.",
        )

    csv_bytes = await batch_csv.read()
    prompt_audio_bytes = await speaker_prompt_audio.read()
    if not prompt_audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="speaker_prompt_audio is required.",
        )

    job_id = create_batch_job(
        csv_bytes=csv_bytes,
        speaker_prompt_audio_bytes=prompt_audio_bytes,
        speaker_prompt_audio_filename=speaker_prompt_audio.filename or "prompt.wav",
        speaker_prompt_text_transcription=prompt_text,
    )
    record_usage(request.app, converted_files=0)
    worker = threading.Thread(
        target=run_batch_job,
        args=(request.app, job_id),
        daemon=True,
    )
    worker.start()
    return serialize_job(job_id, get_job(job_id))


@router.get("/batch/jobs/{job_id}", response_model=BatchJobResponse)
async def get_batch_job_endpoint(request: Request, job_id: str):
    require_api_key(request)
    return serialize_job(job_id, get_job(job_id))


@router.get("/batch/jobs/{job_id}/download")
async def download_batch_job_endpoint(request: Request, job_id: str):
    require_api_key(request)
    job = get_job(job_id)
    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Batch job is not completed yet.",
        )
    zip_path = Path(job["zip_path"])
    if not zip_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch ZIP file is not available.",
        )
    try:
        zip_bytes = zip_path.read_bytes()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read batch ZIP: {exc}",
        ) from exc

    return StreamingResponse(
        BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=breezyvoice-batch.zip"},
    )


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
