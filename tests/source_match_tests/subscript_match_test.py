import unittest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class SubscriptMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Subscript('a', 1)
        string = 'a[1]'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual('a[1]', matcher.GetSource())

    def testAllPartsMatch(self):
        node = create_node.Subscript('a', 1, 2, 3)
        string = 'a[1:2:3]'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual('a[1:2:3]', matcher.GetSource())

    def testSeparatedWithStrings(self):
        node = create_node.Subscript('a', 1, 2, 3)
        string = 'a [ 1 : 2 : 3 ]'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual('a [ 1 : 2 : 3 ]', matcher.GetSource())

    def testSubscriptModule7Partial(self):
        string =  """f"k[a][:-1]" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testSubscriptModule7Partial2(self):
        string =  """f"{k[1:]}" """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testSubscriptModule7Partial3(self):
        string =  """k[1:] """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)