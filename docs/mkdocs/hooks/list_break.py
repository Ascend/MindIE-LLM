# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.


def on_page_markdown(markdown, **kwargs):
    """Insert list breaks before headings that follow list-item admonitions.

    When an admonition (converted from > [!TYPE] by github_admonition.py)
    lives inside a list item, Python-Markdown may fail to close the <li>
    when it encounters a ## heading.  We detect that case and insert a
    column-0 <!-- --> comment to force the list to close.
    """

    lines = markdown.split("\n")
    result = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # A ## heading preceded by <!-- --> means the previous admonition
        # is inside a list item and the parser may not break the list.
        if stripped.startswith("## "):
            # Look back through recent non-blank lines.
            seen_comment = False
            for j in range(i - 1, max(0, i - 20), -1):
                prev = lines[j].strip()
                if prev == "<!-- -->":
                    seen_comment = True
                    break
                if prev and not prev.startswith("<!--"):
                    # hit real content — stop looking
                    break

            if seen_comment:
                result.append("<!-- -->")
                result.append("")

        result.append(line)

    return "\n".join(result)
