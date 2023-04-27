import unittest

import create_node
from fun_with_ast.dynamic_matcher import GetDynamicMatcher


class JoinStrMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.JoinedStr([create_node.Str('fun-with-ast')])
        string = "f'fun-with-ast'"
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
