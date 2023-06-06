import unittest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher


class SetMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Set('c', 'a', 'b')
        string = '{c, a, b}'
        matcher = GetDynamicMatcher(node)
        matcher._match(string)
        self.assertEqual(string, matcher.GetSource())
