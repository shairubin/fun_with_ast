import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class FunctionDefWithDocStringMatcherTest(BaseTestUtils):

    def testNotRealDocString(self):
        string = """def test_fun():\n 4\n"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testNotRealDocString2(self):
        string = """def test_fun():\n '4'\n"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testNotRealDocString3(self):
        string = """def test_fun():\n \"\"\"4\"\"\"\n"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testNotRealDocString4(self):
        string = """def test_fun():\n \"\"\"4\"\"\"\n"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testNotRealDocString5(self):
        string = """def test_fun():\n \"\"\"4\"\"\"\n pass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testNotRealDocString6(self):
        string = """def test_fun():\n \"\"\"4\n4\"\"\"\n pass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testNotRealDocString7(self):
        string = """def test_fun():\n \"\"\"4\n4\n\"\"\"\n pass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNotRealDocString8(self):
        string = """def from_journal(cls, other: "Journal") -> "Journal":
    \"\"\"Creates a new journal by copying configuration and entries from
    another journal object\"\"\"
    new_journal = cls(other.name, **other.config)
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNotRealDocString9(self):
        string = """def from_journal(cls, other: "Journal") -> "Journal":
    \"\"\"Creates
    object\"\"\"
    new_journal = cls(other.name, **other.config)
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNotRealDocString10(self):
        string = """def from_journal(cls, other: "Journal") -> "Journal":
    \"\"\"4\n4\"\"\"
    new_journal = cls(other.name, **other.config)
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    @pytest.mark.skip(reason=" space between ')' and ':' issue TBD")
    def testNotRealDocString10_1(self):
        string = """def foo(a: int) :
    pass
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNotRealDocString10_2(self):
        string = """def foo(a: "int"):
    pass
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testNotRealDocString10_3(self):
        string = """def foo(a:"int"):
    pass
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNotRealDocString11(self):
        string = """def from_journal(cls, other: Journal) -> Journal:
    \"\"\"4\n4\"\"\"
    new_journal = cls(other.name, **other.config)
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
