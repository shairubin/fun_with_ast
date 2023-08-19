import unittest

import pytest
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class AssignMatcherTest(BaseTestUtils):

    def testBasicMatchAssignHexWithUpper(self):
        node = create_node.Assign('a', create_node.Num('0x1F')) # not implemented matching to 0x1f
        string = 'a=0x1F'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)
    def testBasicMatchAssignHexWithLower(self):
        node = create_node.Assign('a', create_node.Num('0x1f'))
        string = 'a=0x1f'
        self._assert_matched_source(node, string)

    def testBasicMatchAssignNone(self):
        node = create_node.Assign('a', create_node.CreateNone('None'))
        string = 'a = \t None # a is None'
        self._assert_matched_source(node, string)

    def testBasicNoMatchAssignNone(self):
        node = create_node.Assign('a', create_node.CreateNone('None'))
        string = 'a = \t none # a is None'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._assert_matched_source(node, string)

    def testBasicMatchAssignString(self):
        node = create_node.Assign('a', create_node.Constant('1', "'"))
        string = "a='1'"
        self._assert_matched_source(node, string)

    def testBasicNoMatchAssignStringWithDoubleQuote(self):
        node = create_node.Assign('a', create_node.Constant('1', "'"))
        string = "a=\"1\""
        self._assert_matched_source(node, string)

    def testBasicMatchAssignStringWithDoubleQuote2(self):
        node = create_node.Assign('a', create_node.Constant('1', "\""))
        string = "a=\"1\""
        self._assert_matched_source(node, string)

    def testBasicMatchAssignString2(self):
        node = create_node.Assign('a', create_node.Constant('12', "\'"))
        string = "a='1''2'"
        self._assert_matched_source(node, string)

    def testBasicMatchAssignTrailingWS(self):
        node = create_node.Assign('a', create_node.Num('1'))
        string = 'a=1 '
        self._assert_matched_source(node, string)

    def testBasicMatchAssign(self):
        node = create_node.Assign('a', create_node.Num('1'))
        string = 'a=1'
        self._assert_matched_source(node, string)

    def testBasicMatchAssignWithNL(self):
        node = create_node.Assign('a', create_node.Num('2'))
        string = 'a=2'
        self._assert_matched_source(node, string)

    def testBasicMatchAssignWithWSAndTab(self):
        node = create_node.Assign('a', create_node.Num('1'))
        string = ' a  =  1  \t'
        self._assert_matched_source(node, string)

    def testMatchMultiAssign(self):
        node = create_node.Assign(['a', 'b'], create_node.Num('2'))
        string = 'a=b=1'
        matcher = GetDynamicMatcher(node)
        matched_string = matcher.GetSource()
        self.assertNotEqual(string, matched_string)

    def testNotMatchMultiAssign(self):
        node = create_node.Assign(['a', 'b'], create_node.Num('1'))
        string = 'a=c=1'
        matcher = GetDynamicMatcher(node)
        matched_string = matcher.GetSource()
        self.assertNotEqual(string, matched_string)

    def testNotMatchMultiAssign2(self):
        node = create_node.Assign(['a', 'b'], create_node.Num('1'))
        string = 'a=c=1\n'
        matcher = GetDynamicMatcher(node)
        matched_string = matcher.GetSource()
        self.assertNotEqual(string, matched_string)


    def testMatchMultiAssignWithWS(self):
        node = create_node.Assign(['a', 'b'], create_node.Num('0o7654'))
        string = 'a\t=\t     b \t  =0o7654 \t'
        self._assert_matched_source(node, string)

    def testMatchMultiAssignWithWSAndComment(self):
        node = create_node.Assign(['a', 'b'], create_node.Num('1'))
        string = 'a\t=\t     b \t  =1 \t #comment'
        self._assert_matched_source(node, string)

    def testMatchMultiAssignNameWithWSAndComment(self):
        node = create_node.Assign(['a', 'b'], create_node.Name('c'))
        string = 'a\t=\t     b \t  =c \t #comment'
        self._assert_matched_source(node, string)

    def testMatchMultiAssignNameWithWSAndComment3(self):
        node = create_node.Assign(['a', 'b'], create_node.Name('c'))
        string = 'a\t=\t     b \t  =c \t #########'
        self._assert_matched_source(node, string)

    def testNotMatchMultiAssignWithWS(self):
        node = create_node.Assign(['a', 'b'], create_node.Num('1'))
        string = 'a\t=\t     bb \t  =1 \t'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)


    def _assert_matched_source(self, node, string):
        self._verify_match(node, string)
