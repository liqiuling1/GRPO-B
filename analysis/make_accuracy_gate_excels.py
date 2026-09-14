import csv
import math
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis"


WORKBOOKS = [
    (
        "accuracy_gate_0.375-0.625_merged_bucket_accepted_difficulty.xlsx",
        [
            (
                "0.375-0.625",
                OUT_DIR / "accuracy_gate_0.375-0.625_merged_bucket_accepted_difficulty_compact.csv",
            )
        ],
    ),
    (
        "accuracy_gate_0.5_merged_bucket_accepted_difficulty.xlsx",
        [
            (
                "0.5",
                OUT_DIR / "accuracy_gate_0.5_merged_bucket_accepted_difficulty_compact.csv",
            )
        ],
    ),
    (
        "accuracy_gate_0.4375-0.5625_merged_bucket_accepted_difficulty.xlsx",
        [
            (
                "1st merged",
                OUT_DIR / "accuracy_gate_0.4375-0.5625_1st_merged_bucket_accepted_difficulty_compact.csv",
            ),
            (
                "2nd merged",
                OUT_DIR / "accuracy_gate_0.4375-0.5625_2nd_merged_bucket_accepted_difficulty_compact.csv",
            ),
        ],
    ),
]


NUMBER_RE = re.compile(r"^-?(?:\d+|\d+\.\d+|\.\d+)(?:[eE][+-]?\d+)?$")


def column_name(index):
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def is_number(value):
    if not value or not NUMBER_RE.match(value):
        return False
    try:
        number = float(value)
    except ValueError:
        return False
    return math.isfinite(number)


def cell_xml(row_index, col_index, value):
    ref = f"{column_name(col_index)}{row_index}"
    value = "" if value is None else str(value)
    if is_number(value):
        return f'<c r="{ref}"><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'


def sheet_xml(rows):
    max_cols = max((len(row) for row in rows), default=1)
    cols = "".join(f'<col min="{i}" max="{i}" width="18" customWidth="1"/>' for i in range(1, max_cols + 1))
    row_xml = []
    for row_index, row in enumerate(rows, 1):
        cells = "".join(cell_xml(row_index, col_index, value) for col_index, value in enumerate(row, 1))
        row_xml.append(f'<row r="{row_index}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"{cols}<sheetData>{''.join(row_xml)}</sheetData>"
        "</worksheet>"
    )


def workbook_xml(sheets):
    sheet_entries = []
    for index, (name, _) in enumerate(sheets, 1):
        safe_name = escape(name[:31])
        sheet_entries.append(f'<sheet name="{safe_name}" sheetId="{index}" r:id="rId{index}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{''.join(sheet_entries)}</sheets>"
        "</workbook>"
    )


def workbook_rels_xml(sheets):
    rels = []
    for index, _ in enumerate(sheets, 1):
        rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(rels)}</Relationships>"
    )


def root_rels_xml():
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def content_types_xml(sheets):
    sheet_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index, _ in enumerate(sheets, 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{sheet_overrides}</Types>"
    )


def write_workbook(filename, sheets):
    output_path = OUT_DIR / filename
    rows_by_sheet = [(name, read_csv(path)) for name, path in sheets]
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml(rows_by_sheet))
        archive.writestr("_rels/.rels", root_rels_xml())
        archive.writestr("xl/workbook.xml", workbook_xml(rows_by_sheet))
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(rows_by_sheet))
        for index, (_, rows) in enumerate(rows_by_sheet, 1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", sheet_xml(rows))
    return output_path


def main():
    for _, sheets in WORKBOOKS:
        for _, path in sheets:
            if not path.exists():
                raise FileNotFoundError(path)

    for filename, sheets in WORKBOOKS:
        print(write_workbook(filename, sheets))


if __name__ == "__main__":
    main()
