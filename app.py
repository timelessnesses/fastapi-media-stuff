from calendar import c
from dotenv import load_dotenv

load_dotenv()
import enum
import io
import os
import random
import string
import typing

import aioredis
import fastapi
import magic
import pydub
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from PIL import Image
from sanitize_filename import sanitize

application = fastapi.FastAPI()
app = application  # uvicorn

import math
import os
import typing
from io import BytesIO

import aiohttp
import discord
from PIL import Image, ImageDraw, ImageFont

class Enum_Status(enum.Enum):
    """
    Discord user status
    """
    online = "online"
    offline = "offline"
    idle = "idle"
    streaming = "streaming"
    do_not_disturb = "dnd"

class Generator:
    def __init__(self):
        self.default_bg = os.path.join(os.path.dirname(__file__), "assets", "card.png")
        self.online = os.path.join(os.path.dirname(__file__), "assets", "online.png")
        self.offline = os.path.join(os.path.dirname(__file__), "assets", "offline.png")
        self.idle = os.path.join(os.path.dirname(__file__), "assets", "idle.png")
        self.dnd = os.path.join(os.path.dirname(__file__), "assets", "dnd.png")
        self.streaming = os.path.join(
            os.path.dirname(__file__), "assets", "streaming.png"
        )
        self.font1 = os.path.join(os.path.dirname(__file__), "assets", "font.ttf")

    async def generate_profile(
        self,
        bg_image: str = None,
        profile_image: str = None,
        level: int = 1,
        user_xp: int = 20,
        next_xp: int = 100,
        server_position: int = 1,
        user_name: str = "Dummy#0000",
        user_status: Enum_Status = "online",
        font_color: typing.Union[typing.List, typing.Tuple, typing.Set] = (
            255,
            255,
            255,
        ),
    ) -> io.BytesIO:
        level += 1  # i hate you rgbcube
        current_xp = 0
        if not bg_image:
            card = Image.open(self.default_bg).convert("RGBA")
        else:
            async with aiohttp.ClientSession() as session:
                async with session.get(bg_image) as r:
                    card = Image.open(BytesIO(await r.read())).convert("RGBA")

            width, height = card.size
            if width == 900 and height == 238:
                pass
            else:
                x1 = 0
                y1 = 0
                x2 = width
                nh = math.ceil(width * 0.264444)
                y2 = 0

                if nh < height:
                    y1 = (height / 2) - 119
                    y2 = nh + y1

                card = card.crop((x1, y1, x2, y2)).resize((900, 238))
        async with aiohttp.ClientSession() as session:
            async with session.get(profile_image) as r:
                profile = Image.open(BytesIO(await r.read())).convert("RGBA")
        profile = profile.resize((180, 180))
        
        user_status = user_status.value.lower()
        
        if user_status == "online":
            status = Image.open(self.online)
        if user_status == "offline":
            status = Image.open(self.offline)
        if user_status == "idle":
            status = Image.open(self.idle)
        if user_status == "streaming":
            status = Image.open(self.streaming)
        if user_status == "dnd":
            status = Image.open(self.dnd)

        status = status.convert("RGBA").resize((40, 40))

        profile_pic_holder = Image.new(
            "RGBA", card.size, (255, 255, 255, 0)
        )  # Is used for a blank image so that i can mask

        # Mask to crop image
        mask = Image.new("RGBA", card.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse(
            (29, 29, 209, 209), fill=(255, 25, 255, 255)
        )  # The part need to be cropped

        # Editing stuff here

        # ======== Fonts to use =============
        font_normal = ImageFont.truetype(self.font1, 36)
        font_small = ImageFont.truetype(self.font1, 20)
        level_font = ImageFont.truetype(self.font1, 30)
        # ======== Colors ========================

        def get_str(xp):
            if xp < 1000:
                return str(xp)
            if xp >= 1000 and xp < 1000000:
                return str(round(xp / 1000, 1)) + "k"
            if xp > 1000000:
                return str(round(xp / 1000000, 1)) + "M"

        draw = ImageDraw.Draw(card)
        draw.text((245, 22), user_name, font_color, font=font_normal)
        draw.text(
            (245, 123),
            f"Server Rank #{server_position}",
            font_color,
            font=font_small,
        )
        draw.text((245, 74), f"Level {level}", font_color, font=level_font)
        draw.text(
            (245, 150),
            f"Exp {get_str(user_xp)}/{get_str(next_xp)}",
            font_color,
            font=font_small,
        )

        # Adding another blank layer for the progress bar
        # Because drawing on card dont make their background transparent
        blank = Image.new("RGBA", card.size, (255, 255, 255, 0))
        blank_draw = ImageDraw.Draw(blank)
        blank_draw.rectangle(
            (245, 185, 750, 205), fill=(255, 255, 255, 0), outline=font_color
        )

        xpneed = next_xp - current_xp
        xphave = user_xp - current_xp

        current_percentage = (xphave / xpneed) * 100
        length_of_bar = (current_percentage * 4.9) + 248
        blank_draw.text(
            (750, 150), f"{round(current_percentage,2)}%", font_color, font=font_small
        )
        blank_draw.rectangle((248, 188, length_of_bar, 202), fill=font_color)
        blank_draw.ellipse(
            (20, 20, 218, 218), fill=(255, 255, 255, 0), outline=font_color
        )

        profile_pic_holder.paste(profile, (29, 29, 209, 209))

        pre = Image.composite(profile_pic_holder, card, mask)
        pre = Image.alpha_composite(pre, blank)

        # Status badge
        # Another blank
        blank = Image.new("RGBA", pre.size, (255, 255, 255, 0))
        blank.paste(status, (169, 169))

        final = Image.alpha_composite(pre, blank)
        final_bytes = BytesIO()
        final.save(final_bytes, "png")
        final_bytes.seek(0)
        return final_bytes


generate_profile = Generator().generate_profile

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
@cache(expire=60)
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


@app.put("/convert_file")
@cache(expire=60)
async def convert_file(
    request: fastapi.Request,
    file: fastapi.UploadFile,
    ext: typing.Optional[str] = "mp4",
):
    """
    convert file to whatever you want
    """
    file = await file.read()
    file_name = sanitize(file.filename)
    with open(file_name, "wb") as f:
        f.write(file)
    os.system(
        f"ffmpeg -i {file_name} -c:v libx264 -c:a aac -strict -2 {file_name}.{ext}"
    )
    with open(f"{file_name}.{ext}", "rb") as f:
        content = io.BytesIO(f.read())
    os.remove(f"{file_name}.{ext}")
    return fastapi.responses.StreamingResponse(
        content=content, media_type=magic.from_buffer(content.read(1024), mime=True)
    )

@app.put("/merge_audio")
@cache(expire=60)
async def merge_audio(
    request: fastapi.Request,
    files: typing.List[fastapi.UploadFile],
):
    merged = pydub.AudioSegment.empty()
    for file in files:
        merged += pydub.AudioSegment.from_file(io.BytesIO(await file.read()))
    content = io.BytesIO()
    content.write(merged.raw_data)
    content.seek(0)
    return fastapi.responses.StreamingResponse(
        content=content, media_type=magic.from_buffer(content.read(1024), mime=True)
    )

@app.put("/slice_audio")
async def slice_audio(
    request: fastapi.Request,
    file: fastapi.UploadFile,
    start: int,
    end: int,
):
    audio = pydub.AudioSegment.from_file(io.BytesIO(await file.read()))
    sliced = audio[start:end]
    content = io.BytesIO()
    content.write(sliced.raw_data)
    content.seek(0)
    return fastapi.responses.StreamingResponse(
        content=content, media_type=magic.from_buffer(content.read(1024), mime=True)
    )
    
@app.get("/level_image_generator")
async def level_image_generator(
    request: fastapi.Request,
    bg_image: str=None,
    profile_image: str=None,
    level: int=1,
    user_xp: int=0,
    next_xp: int=100,
    server_position: int=1,
    user_name: str = "Dummy#0000",
    user_status: Enum_Status = Enum_Status.online,
    font_color: str = "255,255,255",
):
    return fastapi.responses.StreamingResponse(
        generate_profile(
            bg_image=bg_image,
            profile_image=profile_image,
            level=level,
            user_xp=user_xp,
            next_xp=next_xp,
            server_position=server_position,
            user_name=user_name,
            user_status=user_status,
            font_color=font_color.split(","),
        ),
        media_type="image/png",
    )

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url(
        os.environ["REDIS_URI"], encoding="utf8", decode_responses=True
    )
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")


if __name__ == "__main__":
    app.run("localhost", 5000)
