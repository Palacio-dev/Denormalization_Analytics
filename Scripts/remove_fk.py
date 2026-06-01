import re
from pathlib import Path


def extract_foreign_keys(sql_text: str) -> dict[str, set[str]]:
    """
    Extract FK columns from ALTER TABLE statements.

    Returns:
        {
            "consumo_anual": {"id_paises", "tipo_energia"},
            "emissao_g_efeito_estufa": {"id_paises"},
            ...
        }
    """

    fk_pattern = re.compile(
        r"ALTER\s+TABLE\s+IF\s+EXISTS\s+"
        r"(?:\w+\.)?(\w+)\s+"
        r"ADD\s+CONSTRAINT\s+\w+\s+"
        r"FOREIGN\s+KEY\s*\(\s*(\w+)\s*\)",
        re.IGNORECASE | re.DOTALL,
    )

    foreign_keys = {}

    for table_name, column_name in fk_pattern.findall(sql_text):
        foreign_keys.setdefault(table_name, set()).add(column_name)

    return foreign_keys


def remove_fk_columns_from_create_tables(
    sql_text: str,
    foreign_keys: dict[str, set[str]]
) -> str:
    """
    Remove FK columns from CREATE TABLE blocks.
    """

    create_table_pattern = re.compile(
        r"(CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+"
        r"(?:\w+\.)?(\w+)\s*"
        r"\(.*?\)\s*;)",
        re.IGNORECASE | re.DOTALL,
    )

    def process_create_table(match):
        full_block = match.group(1)
        table_name = match.group(2)

        fk_columns = foreign_keys.get(table_name, set())

        if not fk_columns:
            return full_block

        lines = full_block.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip FK column definitions
            if stripped:
                column_match = re.match(r"^(\w+)\s+", stripped)

                if column_match:
                    column_name = column_match.group(1)

                    if (
                        column_name in fk_columns
                        and not stripped.startswith("CONSTRAINT")
                    ):
                        continue

            cleaned_lines.append(line)

        # Remove trailing comma before constraint or closing parenthesis
        for i in range(len(cleaned_lines) - 1):
            current = cleaned_lines[i].rstrip()
            next_line = cleaned_lines[i + 1].strip()

            if (
                current.endswith(",")
                and (
                    next_line.startswith("CONSTRAINT")
                    or next_line == ");"
                    or next_line == ")"
                )
            ):
                cleaned_lines[i] = current[:-1]

        return "\n".join(cleaned_lines)

    return create_table_pattern.sub(process_create_table, sql_text)





def normalize_schema(input_file: str, output_file: str):
    sql_text = Path(input_file).read_text(encoding="utf-8")

    foreign_keys = extract_foreign_keys(sql_text)

    sql_text = remove_fk_columns_from_create_tables(
        sql_text,
        foreign_keys
    )
    Path(output_file).write_text(sql_text.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    normalize_schema(
        "../Experiment_schemes/Train/ex10.txt",
        "schema_without_fk.txt"
    )