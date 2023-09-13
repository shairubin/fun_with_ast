import unittest

import pytest

from fun_with_ast.manipulate_node import create_node as create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class AugAssignMatcherTest(BaseTestUtils):


    def testBasicMatch(self):
        node = create_node.AugAssign('a', create_node.Add(), create_node.Num(12))
        string = 'a+=1\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testNotMatchWithVarAndTab(self):
        node = create_node.AugAssign('a', create_node.Add(), create_node.Name('c'))
        string = '       \t        a += b\n'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)
        #self.assertNotEqual(string, matcher.GetSource())

    def testMatchWithVarAndTab(self):
        node = create_node.AugAssign('a', create_node.Add(), create_node.Name('b'))
        string = '       \t        a += b\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithVarAndTab2(self):
        node = create_node.AugAssign('a', create_node.Add(), create_node.Name('b'))
        string = '               a +=\tb\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

########################################################################
# from input tests
########################################################################

    def testFromInput(self):
        string = """
attn_weights -= a()
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFromInput1(self):
        string = """a += a()"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFromInput11(self):
        string = """a += a()\n"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testFromInput12(self):
        string = """a += 1\n"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testFromInput13(self):
        string = """a += 'str'\n"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFromInput14(self):
        string = """a+='str'"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
