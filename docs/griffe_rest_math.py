"""Griffe extension: render the reST math of the docstrings with MathJax.

The package docstrings follow the NumPy/Sphinx convention (``:math:`...``` roles and
``.. math::`` blocks). mkdocstrings renders docstrings as Markdown, so this extension
rewrites them into the ``$...$`` / ``$$...$$`` delimiters understood by
``pymdownx.arithmatex`` and turns ``.. code-block::`` directives into fenced code.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any

import griffe

_ROLE = re.compile(r":math:`([^`]+)`")


def _convert_blocks(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        directive = re.match(r"\.\. (math|code-block)::\s*(.*)$", stripped)
        if directive is None:
            out.append(line)
            i += 1
            continue
        kind, rest = directive.groups()
        body: list[str] = []
        i += 1
        while i < len(lines) and (not lines[i].strip() or lines[i].startswith(indent + " ")):
            body.append(lines[i])
            i += 1
        while body and not body[-1].strip():
            body.pop()
        content = textwrap.dedent("\n".join(body)).strip("\n")
        if kind == "math":
            math = (rest + "\n" + content).strip() if rest else content
            out += [f"{indent}$$", *(indent + m for m in math.splitlines()), f"{indent}$$", ""]
        else:
            out += [f"{indent}```{rest.strip()}", *(indent + c for c in content.splitlines()),
                    f"{indent}```", ""]  # fmt: skip
    return "\n".join(out)


def convert(text: str) -> str:
    """Convert reST math roles/blocks and code-block directives to Markdown."""
    return _ROLE.sub(lambda m: f"${m.group(1)}$", _convert_blocks(text))


class RestMath(griffe.Extension):
    """Rewrite every docstring once the object has been loaded."""

    def on_object(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        if obj.docstring is not None:
            obj.docstring.value = convert(obj.docstring.value)
