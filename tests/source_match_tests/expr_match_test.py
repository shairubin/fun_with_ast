import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ExprMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        string2 = 'a.b()\n'
        call_node = GetNodeFromInput(string2)
        matcher2 = GetDynamicMatcher(call_node)
        matcher2._match(string2)
        source2 = matcher2.GetSource()
        self.assertEqual(source2, string2)

    def testBasicMatch2(self):
        node = create_node.Call('a.b')
        string = 'a.b()\n '
        expr_node = create_node.Expr(node)
        matcher = GetDynamicMatcher(expr_node)
        matcher.do_match(string)

    def testBasicMatchWS(self):
        node = create_node.Call('a.b')
        string = ' a.b()\n '
        expr_node = create_node.Expr(node)
        matcher = GetDynamicMatcher(expr_node)
        matcher._match(string)

    def testSimpleExpr(self):
        string = '4\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testSimpleExpr2(self):
        string = '4'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testSimpleExpr3(self):
        string = 'a'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testSimpleExpr4(self):
        string = "'a'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testSimpleExpr5(self):
        string = "\"b\""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testSimpleExpr6(self):
        string = "\"\"\"b\"\"\""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSimpleExpr7(self):
        string = "\"b\"\n # comment"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testSimpleExpr8(self):
        string = "'7'\n   " # without a module node, this should fail
        node = GetNodeFromInput(string)
        with pytest.raises(AssertionError):
            self._verify_match(node, string)

    def testSimpleExpr9(self):
        string = "'7'\n   "
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
