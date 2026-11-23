
from pathlib import Path
import csv
import json

import pymupdf
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation
from bs4 import BeautifulSoup


def load_pdf(file_path):
    results = []

    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                results.append({
                    "text": text,
                    "metadata": {
                        "source": Path(file_path).name,
                        "file_type": "pdf",
                        "page": page_number
                    }
                })

    return results


def load_docx(file_path):
    document = Document(file_path)
    results = []

    for index, paragraph in enumerate(document.paragraphs, start=1):
        text = paragraph.text.strip()

        if text:
            results.append({
                "text": text,
                "metadata": {
                    "source": Path(file_path).name,
                    "file_type": "docx",
                    "paragraph": index
                }
            })

    for table_number, table in enumerate(document.tables, start=1):
        for row_number, row in enumerate(table.rows, start=1):
            text = " | ".join(
                cell.text.strip() for cell in row.cells
            )

            if text.strip():
                results.append({
                    "text": text,
                    "metadata": {
                        "source": Path(file_path).name,
                        "file_type": "docx",
                        "table": table_number,
                        "row": row_number
                    }
                })

    return results


def load_xlsx(file_path):
    workbook = load_workbook(
        file_path,
        read_only=True,
        data_only=True
    )
    results = []

    try:
        for sheet in workbook.worksheets:
            rows = sheet.iter_rows(values_only=True)
            headers = next(rows, None)

            for row_number, row in enumerate(rows, start=2):
                values = []

                for index, value in enumerate(row):
                    if value is not None:
                        header = (
                            str(headers[index])
                            if headers
                            and index < len(headers)
                            and headers[index] is not None
                            else f"Column_{index + 1}"
                        )
                        values.append(f"{header}: {value}")

                text = " | ".join(values)

                if text:
                    results.append({
                        "text": text,
                        "metadata": {
                            "source": Path(file_path).name,
                            "file_type": "xlsx",
                            "sheet": sheet.title,
                            "row": row_number
                        }
                    })
    finally:
        workbook.close()

    return results


def load_pptx(file_path):
    presentation = Presentation(file_path)
    results = []

    for slide_number, slide in enumerate(
        presentation.slides, start=1
    ):
        slide_text = []

        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text.strip()
                if text:
                    slide_text.append(text)

            if shape.has_table:
                for row in shape.table.rows:
                    values = [
                        cell.text.strip() for cell in row.cells
                    ]
                    slide_text.append(" | ".join(values))

        combined_text = "\n".join(slide_text).strip()

        if combined_text:
            results.append({
                "text": combined_text,
                "metadata": {
                    "source": Path(file_path).name,
                    "file_type": "pptx",
                    "slide": slide_number
                }
            })

    return results


def load_text(file_path):
    text = Path(file_path).read_text(
        encoding="utf-8-sig",
        errors="replace"
    ).strip()

    if not text:
        return []

    return [{
        "text": text,
        "metadata": {
            "source": Path(file_path).name,
            "file_type": Path(file_path).suffix.lower().lstrip(".")
        }
    }]


def load_csv(file_path):
    results = []

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=2):
            text = " | ".join(
                f"{key}: {value}"
                for key, value in row.items()
                if key and value is not None and value.strip()
            )

            if text:
                results.append({
                    "text": text,
                    "metadata": {
                        "source": Path(file_path).name,
                        "file_type": "csv",
                        "row": row_number
                    }
                })

    return results


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as file:
        data = json.load(file)

    text = json.dumps(data, ensure_ascii=False, indent=2)

    if not text.strip():
        return []

    return [{
        "text": text,
        "metadata": {
            "source": Path(file_path).name,
            "file_type": "json"
        }
    }]


def load_html(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as file:
        soup = BeautifulSoup(file, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = soup.get_text(separator="\n", strip=True)

    if not text:
        return []

    return [{
        "text": text,
        "metadata": {
            "source": Path(file_path).name,
            "file_type": "html"
        }
    }]


def load_document(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    extension = path.suffix.lower()

    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".xlsx": load_xlsx,
        ".pptx": load_pptx,
        ".txt": load_text,
        ".md": load_text,
        ".csv": load_csv,
        ".json": load_json,
        ".html": load_html,
        ".htm": load_html
    }

    if extension not in loaders:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    return loaders[extension](path)
