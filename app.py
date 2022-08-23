import enum
import io
import typing

import fastapi
import moviepy
import pydub
from PIL import Image

application = fastapi.FastAPI()
app = application  # gunicorn


linear_enum = {
    "nearest": Image.NEAREST,
    "lanczos": Image.LANCZOS,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
}


class linear_enum_typing(enum.Enum):
    nearest = "nearest"
    lanczos = "lanczos"
    bilinear = "bilinear"
    bicubic = "bicubic"


@app.post("/resize", response_class=fastapi.responses.StreamingResponse)
async def resize(
    request: fastapi.Request,
    image: fastapi.UploadFile,
    width: int,
    height: int,
    resample: typing.Optional[linear_enum_typing] = linear_enum_typing.lanczos,
    ext: typing.Optional[str] = "png",
) -> bytes:
    """
    resize cutely
    needed image in body request or in form tag named image :3c
    """
    try:
        form = await request.form()
        con = await form["image"].read()
        image = Image.open(io.BytesIO(con))
    except (KeyError, AttributeError):
        image = Image.open(io.BytesIO(await request.body()))
    # except Exception as e:
    # return f"{e.__class__}: {str(e)}", 500
    resample = linear_enum.get(resample)
    image = image.resize((width, height), resample=resample)
    image_edited = io.BytesIO()
    image.save(image_edited, format="png")
    image_edited.seek(0)
    return fastapi.responses.StreamingResponse(
        content=image_edited, media_type=f"image/{ext}"
    )


@app.get("/", response_class=fastapi.responses.HTMLResponse)
async def root():
    return """
        <h1>Resize image api</h1>
        <p>usage:<br>provide param width/height/resampling (default lanczos) (optional) (look at resampling header for every value possible) and ext(default png) (optional) <br>you need to upload the image to request body or feed it as a file form under tag name image<h2>resampling</h2>resampling if you want different resizing results<br>- nearest: use nearest neighbors<br>- bilinear: linear interpolation<br>- bicubic: cubic spline interpolation<br>- lanczos: a high-quality downsampling filter (default)<br><a href="/docs">click me to go to more of detailed api info</a>
        """


if __name__ == "__main__":
    app.run("localhost", 5000)
