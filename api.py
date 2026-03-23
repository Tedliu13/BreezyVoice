# OpenAI API Spec. Reference: https://platform.openai.com/docs/api-reference/audio/createSpeech

import csv
import os
import re
import tempfile
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

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


class SpeechRequest(BaseModel):
    model: str = ""
    input: str = Field(
        description="The content that will be synthesized into speech. You can include phonetic symbols if needed, though they should be used sparingly.",
        examples=["今天天氣真好"],
    )
    response_format: str = ""
    speed: float = 1.0


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    app.state.cosyvoice = CustomCosyVoice(app.state.settings.model_path)
    app.state.bopomofo_converter = G2PWConverter()
    app.state.prompt_speech_16k = load_wav(
        app.state.settings.speaker_prompt_audio_path, 16000
    )
    yield
    del app.state.cosyvoice
    del app.state.bopomofo_converter


app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/v1")
WEB_DIR = Path(__file__).resolve().parent / "web"


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(WEB_DIR / "index.html")


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
    return StreamingResponse(
        BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


@router.post("/batch/speech")
async def batch_speech_endpoint(
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
        zip_file.writestr("manifest.csv", "\n".join(manifest_rows).encode("utf-8"))

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=breezyvoice-batch.zip"},
    )


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
