import os
import requests
import pydantic

def main(book_id:str) -> None:
    _reset_chapter_html()
    book_resp = _api_v1_book(book_id)
    for chapter_api_url in book_resp.chapters:
        chapter_resp = _api_v1_book_chapter(chapter_api_url)
        _api_v2_epubs(chapter_resp.content, chapter_resp.filename)

def _reset_chapter_html() -> None:
    # delete all files in chapter_html
    for file in os.listdir("chapter_html"):
        os.remove(f"chapter_html/{file}")

class APIV1BookResponse(pydantic.BaseModel):
    url: pydantic.HttpUrl
    chapters: list[pydantic.HttpUrl]
    cover: pydantic.HttpUrl
    web_url: pydantic.HttpUrl
    description: str

def _api_v1_book(book_id:str) -> APIV1BookResponse:
    url = f"https://learning.oreilly.com/api/v1/book/{book_id}/"
    resp = requests.get(url)
    resp.raise_for_status()
    return APIV1BookResponse(**resp.json())

class APIV1BookChapterResponse(pydantic.BaseModel):
    filename: str
    content: pydantic.HttpUrl

def _api_v1_book_chapter(chapter_api_url:pydantic.HttpUrl) -> APIV1BookChapterResponse:
    resp = requests.get(chapter_api_url)
    resp.raise_for_status()
    return APIV1BookChapterResponse(**resp.json())

def _api_v2_epubs(chapter_file_url:pydantic.HttpUrl, filename:str) -> None:
    resp = requests.get(chapter_file_url)
    resp.raise_for_status()
    with open(f"chapter_html/{filename}", "wb") as f:
        f.write(resp.content)

if __name__ == "__main__":
    main("9781098156664")
