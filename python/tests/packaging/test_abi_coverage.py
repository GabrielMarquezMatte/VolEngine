import importlib.util
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPT = REPOSITORY / "cpp" / "scripts" / "check_abi_coverage.py"
spec = importlib.util.spec_from_file_location("check_abi_coverage", SCRIPT)
assert spec is not None and spec.loader is not None
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)

HEADER = """
#define VE_API __attribute__((visibility("default")))
VE_API int32_t ve_abi_version(void);
/* VE_API void ve_commented_out(void); */
VE_API ve_status ve_batch_set_jobs(ve_batch* b, const size_t* job_spec, const size_t* job_series,
                                    size_t n_jobs);
VE_API void ve_x_free(ve_x* x);
"""


def test_declared_finds_every_api_function():
    assert coverage.declared(HEADER) == {"ve_abi_version", "ve_batch_set_jobs", "ve_x_free"}


def test_used_ignores_comments_and_strings():
    text = '// ve_fake(1);\n/* ve_other(2) */\nauto s = "ve_string(3)";\nve_real(4);\nauto f = ve_pointer;\n'
    assert coverage.used(text) == {"ve_real", "ve_pointer"}


def test_unused_function_outside_the_allowed_list_fails(tmp_path, capsys):
    header = tmp_path / "volengine.h"
    header.write_text(HEADER)
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    (wrapper / "a.hpp").write_text("void f() { ve_abi_version(); ve_x_free(nullptr); }")
    assert coverage.main(["--header", str(header), "--wrapper", str(wrapper)]) == 1
    assert "ve_batch_set_jobs" in capsys.readouterr().out


def test_every_function_used_passes(tmp_path):
    header = tmp_path / "volengine.h"
    header.write_text(HEADER)
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    (wrapper / "a.hpp").write_text("void f() { ve_abi_version(); ve_x_free(0); g(ve_batch_set_jobs); }")
    assert coverage.main(["--header", str(header), "--wrapper", str(wrapper)]) == 0


def test_real_header_is_covered_by_the_real_wrapper():
    assert coverage.main([]) == 0


def test_allowed_list_only_names_declared_functions():
    header = (REPOSITORY / "include" / "volengine" / "volengine.h").read_text(encoding="utf-8")
    assert coverage.UNUSED_ALLOWED <= coverage.declared(header)
