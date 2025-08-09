# flake8: noqa: F401
from __future__ import annotations

from .pytesseract import ALTONotSupported
from .pytesseract import get_languages
from .pytesseract import get_tesseract_version
from .pytesseract import image_to_alto_xml
from .pytesseract import image_to_boxes
from .pytesseract import image_to_data
from .pytesseract import image_to_osd
from .pytesseract import image_to_pdf_or_hocr
from .pytesseract import image_to_string
from .pytesseract import Output
from .pytesseract import run_and_get_multiple_output
from .pytesseract import run_and_get_output
from .pytesseract import TesseractError
from .pytesseract import TesseractNotFoundError
from .pytesseract import TSVNotSupported
from .pytesseract import set_tesseract_cmd
from .pytesseract import set_tessdata_dir


__version__ = '0.3.14.wt'
