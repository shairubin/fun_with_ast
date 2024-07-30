from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class NamedExprMatcherTest(BaseTestUtils):

    def testSimpleNamedExpr(self):
        string = '(a := 3)'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNamedExpr(self):
        string = '(a :=b)'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNamedExpr_2(self):
        string = '(a := \'x\')'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNamedExpr_3(self):
        string = '(a := a.b.c())'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNamedExpr_4(self):
        string = '(a := a.b.c())'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testNamedExpr_5(self):
        string = '(a := f"{a}a")'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testNamedExprInIf(self):
        string = 'if a:=3:\n  a=7'
        node = GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string)
