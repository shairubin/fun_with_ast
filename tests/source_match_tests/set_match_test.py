import unittest

from manipulate_node import create_node
from fun_with_ast.dynamic_matcher import GetDynamicMatcher


class SetMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Set('c', 'a', 'b')
        string = '{c, a, b}'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
