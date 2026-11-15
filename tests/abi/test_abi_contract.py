"""Unit tests of scripts/abi_contract.py. Run: python -m unittest discover -s tests/abi"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import abi_contract  # noqa: E402


class ContractLines(unittest.TestCase):
    def test_multiline_declaration_is_one_line(self) -> None:
        header = "VE_API ve_status ve_f(const size_t* a,\n    size_t n);"
        self.assertEqual(abi_contract.contract_lines(header), ["function ve_status ve_f(const size_t*, size_t)"])

    def test_define_drops_trailing_comment(self) -> None:
        self.assertEqual(abi_contract.contract_lines("#define VE_X 1 /* note */"), ["define VE_X 1"])

    def test_pointer_to_const_pointer(self) -> None:
        header = "VE_API ve_status ve_g(const ve_spec* const* specs, size_t n_specs);"
        self.assertEqual(
            abi_contract.contract_lines(header), ["function ve_status ve_g(const ve_spec* const*, size_t)"]
        )

    def test_void_parameter_and_pointer_return(self) -> None:
        self.assertEqual(
            abi_contract.contract_lines("VE_API const char* ve_last_error(void);"),
            ["function const char* ve_last_error(void)"],
        )

    def test_typedefs_and_platform_macros(self) -> None:
        header = "#define VE_API __declspec(dllexport)\ntypedef int32_t ve_status;\ntypedef struct ve_context ve_context;"
        self.assertEqual(
            abi_contract.contract_lines(header), ["typedef int32_t ve_status", "typedef struct ve_context ve_context"]
        )

    def test_parameter_names_do_not_matter(self) -> None:
        a = abi_contract.contract_lines("VE_API void ve_h(double* out, size_t count);")
        b = abi_contract.contract_lines("VE_API void ve_h(double* values, size_t n);")
        self.assertEqual(a, b)


class Comparisons(unittest.TestCase):
    def test_removed_line_fails_base(self) -> None:
        self.assertEqual(abi_contract.removed_lines(["define A 1", "define B 2"], ["define A 1"]), ["define B 2"])

    def test_changed_line_fails_base(self) -> None:
        self.assertEqual(abi_contract.removed_lines(["define A 1"], ["define A 2"]), ["define A 1"])

    def test_added_line_passes_base(self) -> None:
        self.assertEqual(abi_contract.removed_lines(["define A 1"], ["define A 1", "define B 2"]), [])


class ExportList(unittest.TestCase):
    CONTRACT = ["define VE_X 1", "function int32_t ve_a(void)", "function void ve_b(ve_x*)"]

    def test_def_lists_functions_only(self) -> None:
        text = abi_contract.def_file(self.CONTRACT, "x.dll")
        self.assertEqual(text.split(), ["LIBRARY", "x.dll", "EXPORTS", "ve_a", "ve_b"])

    def test_linker_exports_must_match(self) -> None:
        linker = "LIBRARY libx.dll\nEXPORTS\n    ve_a @ 1\n    ve_b @ 2\n"
        self.assertEqual(abi_contract.export_differences(self.CONTRACT, linker), [])
        extra = linker + "    ve_c @ 3\n"
        self.assertEqual(abi_contract.export_differences(self.CONTRACT, extra), ["exported but not in the contract: ve_c"])
        missing = "EXPORTS\n    ve_a @ 1\n"
        self.assertEqual(abi_contract.export_differences(self.CONTRACT, missing), ["in the contract but not exported: ve_b"])


class CommittedContract(unittest.TestCase):
    def test_header_matches_committed_contract(self) -> None:
        header = (ROOT / "include" / "volengine" / "volengine.h").read_text(encoding="utf-8")
        committed = (ROOT / "include" / "volengine" / "volengine.abi").read_text(encoding="utf-8")
        self.assertEqual(abi_contract.contract_lines(header), committed.splitlines())

    def test_stable_tags_start_at_one(self) -> None:
        tags = ["v1.0.0-rc.1", "v0.9.0", "v1.0.0", "v1.2.0", "v2.0.0-rc.1"]
        self.assertEqual(abi_contract.latest_stable(tags), "v1.2.0")
        self.assertIsNone(abi_contract.latest_stable(["v0.9.0", "v1.0.0-rc.1"]))

    def test_check_command_reports_a_difference(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            contract = Path(folder) / "volengine.abi"
            contract.write_text("define VE_ABI_VERSION 0\n", encoding="utf-8")
            header = Path(folder) / "volengine.h"
            header.write_text("#define VE_ABI_VERSION 1\n", encoding="utf-8")
            self.assertEqual(abi_contract.main(["check", "--header", str(header), "--contract", str(contract)]), 1)
            self.assertEqual(abi_contract.main(["write", "--header", str(header), "--contract", str(contract)]), 0)
            self.assertEqual(abi_contract.main(["check", "--header", str(header), "--contract", str(contract)]), 0)


if __name__ == "__main__":
    unittest.main()
