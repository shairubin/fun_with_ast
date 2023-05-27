import unittest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.dynamic_matcher import GetDynamicMatcher


class SetMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Set('c', 'a', 'b')
        string = '{c, a, b}'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
