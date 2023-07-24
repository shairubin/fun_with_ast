import unittest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput


class ExprMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        string2 = 'a.b()\n'
        call_node = GetNodeFromInput(string2)
        matcher2 = GetDynamicMatcher(call_node)
        matcher2._match(string2)
        source2 = matcher2.GetSource()
        self.assertEqual(source2, string2)

    def testBasicMatch2(self):
        node = create_node.Call310('a.b')
        string = 'a.b()\n '
        expr_node = create_node.Expr(node)
        matcher = GetDynamicMatcher(expr_node)
        matcher.do_match(string)

    def testBasicMatchWS(self):
        node = create_node.Call310('a.b')
        string = ' a.b()\n '
        expr_node = create_node.Expr(node)
        matcher = GetDynamicMatcher(expr_node)
        matcher._match(string)
