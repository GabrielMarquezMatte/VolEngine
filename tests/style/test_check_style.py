import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_style

CPP_ELSE = "int f(int x)\n{\n    if (x) {\n        return 1;\n    }\n    else {\n        return 2;\n    }\n}\n"
PY_ELIF = "def f(x):\n    if x:\n        return 1\n    elif x < 0:\n        return 2\n    return 3\n"
PY_PARAMS = "def f(a, b, c, d, e, g):\n    return a\n"


def write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def rules_reported(capsys):
    lines = capsys.readouterr().out.splitlines()
    return [line.split(": ")[1] for line in lines if line]


@pytest.mark.parametrize(
    "relative,source,rule",
    [
        ("src/core/a.cpp", CPP_ELSE, "else"),
        ("python/src/pkg/a.py", PY_ELIF, "else"),
        ("python/src/pkg/a.py", PY_PARAMS, "function-params"),
    ],
)
def test_rule_detected(tmp_path, capsys, relative, source, rule):
    write(tmp_path, relative, source)
    assert check_style.main(tmp_path) == 1
    assert rules_reported(capsys) == [rule]


def test_violation_line_has_path_line_rule_message(tmp_path, capsys):
    write(tmp_path, "src/core/a.cpp", CPP_ELSE)
    check_style.main(tmp_path)
    line = capsys.readouterr().out.strip()
    assert line.startswith("src/core/a.cpp:6: else: ")


def test_ternary_and_else_in_string_allowed(tmp_path):
    write(tmp_path, "python/src/pkg/a.py", 'x = 1 if True else 2\ny = "else"\n')
    write(tmp_path, "src/core/a.cpp", 'const char* s = "else"; // else here\n/* else */\n')
    assert check_style.main(tmp_path) == 0


@pytest.mark.parametrize("keyword", ["for", "while", "try"])
def test_python_loop_else_rejected(tmp_path, capsys, keyword):
    sources = {
        "for": "def f(xs):\n    for x in xs:\n        pass\n    else:\n        pass\n",
        "while": "def f(x):\n    while x:\n        pass\n    else:\n        pass\n",
        "try": "def f():\n    try:\n        pass\n    except ValueError:\n        pass\n    else:\n        pass\n",
    }
    write(tmp_path, "python/tests/test_a.py", sources[keyword])
    assert check_style.main(tmp_path) == 1
    assert rules_reported(capsys) == ["else"]


def test_nine_modules_in_folder_rejected(tmp_path, capsys):
    for index in range(9):
        write(tmp_path, f"src/core/m{index}.hpp", "#pragma once\n")
        write(tmp_path, f"src/core/m{index}.cpp", "int x;\n")
    assert check_style.main(tmp_path) == 1
    assert rules_reported(capsys) == ["folder-modules"]


def test_eight_modules_with_pairs_allowed(tmp_path):
    for index in range(8):
        write(tmp_path, f"src/core/m{index}.hpp", "#pragma once\n")
        write(tmp_path, f"src/core/m{index}.cpp", "int x;\n")
    assert check_style.main(tmp_path) == 0


def test_init_py_not_counted_as_module(tmp_path):
    for index in range(8):
        write(tmp_path, f"python/src/pkg/m{index}.py", "x = 1\n")
    write(tmp_path, "python/src/pkg/__init__.py", "")
    assert check_style.main(tmp_path) == 0


def test_long_python_function_rejected(tmp_path, capsys):
    body = "".join(f"    x{index} = {index}\n" for index in range(60))
    write(tmp_path, "python/src/pkg/a.py", "def f():\n" + body)
    assert check_style.main(tmp_path) == 1
    assert rules_reported(capsys) == ["function-length"]


def test_sixty_line_python_function_allowed(tmp_path):
    body = "".join(f"    x{index} = {index}\n" for index in range(59))
    write(tmp_path, "python/src/pkg/a.py", "def f():\n" + body)
    assert check_style.main(tmp_path) == 0


def test_python_function_length_ignores_blank_lines_and_docstring(tmp_path):
    body = "".join(f"    x{index} = {index}\n\n" for index in range(50))
    docstring = '    """one\n    two\n    three\n    four\n    five\n    six\n    seven\n    eight\n    nine\n    ten\n    """\n'
    write(tmp_path, "python/src/pkg/a.py", "def f():\n" + docstring + body + "    return 0\n")
    assert check_style.main(tmp_path) == 0


def test_self_and_cls_not_counted_as_params(tmp_path):
    source = "class A:\n    def f(self, a, b, c, d, e):\n        return a\n"
    write(tmp_path, "python/src/pkg/a.py", source)
    assert check_style.main(tmp_path) == 0


def test_python_nesting_above_three_rejected(tmp_path, capsys):
    source = (
        "def f(xs):\n"
        "    for a in xs:\n"
        "        for b in a:\n"
        "            for c in b:\n"
        "                for d in c:\n"
        "                    pass\n"
    )
    write(tmp_path, "python/src/pkg/a.py", source)
    assert check_style.main(tmp_path) == 1
    assert rules_reported(capsys) == ["nesting"]


def test_python_nesting_of_three_allowed(tmp_path):
    source = "def f(xs):\n    for a in xs:\n        for b in a:\n            for c in b:\n                pass\n"
    write(tmp_path, "python/src/pkg/a.py", source)
    assert check_style.main(tmp_path) == 0


@pytest.mark.parametrize(
    "relative",
    ["build/a.cpp", "python/.venv/a.py", "tests/reference/fixtures/a.py", "tests/reference/data/a.py", "docs/a.py"],
)
def test_ignored_and_unscanned_paths_skipped(tmp_path, relative):
    write(tmp_path, relative, "def f(a, b, c, d, e, g):\n    if a:\n        return 1\n    else:\n        return 2\n")
    assert check_style.main(tmp_path) == 0
