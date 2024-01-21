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
