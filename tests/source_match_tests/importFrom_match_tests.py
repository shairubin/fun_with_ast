from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ImportMatcherTest(BaseTestUtils):

    def testBasicMatchImportFrom(self):
        string = 'from x import a\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testBasicMatchImportFrom2(self):
        string = 'from y.x import a, b\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testBasicMatchImportFrom3(self):
        string = 'from x.y.z import \t a,      b\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicMatchImportFrom4(self):
        string = 'from z import a,b,  c,   \td\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicMatchImportFrom5(self):
        string = 'from z import (a,b,  c,   \td)\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicMatchImportFrom6(self):
        string = 'from z import (a,b,  c,   \td   )     \n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicMatchImportFrom7(self):
        string = """from z import (a,\n     b,c\n)     \n"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
