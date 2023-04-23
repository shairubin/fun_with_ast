import unittest

import pytest

import create_node
import source_match
from dynamic_matcher import GetDynamicMatcher


class AssignMatcherTest(unittest.TestCase):

    def testBasicMatchAssignHex(self):
        node = create_node.Assign('a', create_node.Num(0x1F))
        string = 'a=0x1F'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(NotImplementedError):
            matcher.Match(string)

    def testBasicMatchAssignTrailingWS(self):
        node = create_node.Assign('a', create_node.Num(1))
        string = 'a=1 '
        self._assert_matched_source(node, string)

    def testBasicMatchAssign(self):
        node = create_node.Assign('a', create_node.Num(1))
        string = 'a=1'
        self._assert_matched_source(node, string)

    def testBasicMatchAssignWithNL(self):
        node = create_node.Assign('a', create_node.Num(1))
        string = 'a=1\n'
        self._assert_matched_source(node, string)

    def testBasicMatchAssignWithWSAndTab(self):
        node = create_node.Assign('a', create_node.Num(1))
        string = ' a  =  1  \t'
        self._assert_matched_source(node, string)

    def testBasicMatchAssignWithWSAndTab2(self):
        node = create_node.Assign('a', create_node.Num(1))
        string = ' a  =  1  \t\n'
        self._assert_matched_source(node, string)

    #@pytest.mark.xfail(strict=True)
    def testMatchMultiAssign(self):
        node = create_node.Assign(['a', 'b'], create_node.Num(2))
        string = 'a=b=1'
        matcher = GetDynamicMatcher(node)
        matched_string = matcher.GetSource()
        self.assertNotEqual(string, matched_string)

    def testNotMatchMultiAssign(self):
        node = create_node.Assign(['a', 'b'], create_node.Num(1))
        string = 'a=c=1'
        matcher = GetDynamicMatcher(node)
        matched_string = matcher.GetSource()
        self.assertNotEqual(string, matched_string)

    def testNotMatchMultiAssign2(self):
        node = create_node.Assign(['a', 'b'], create_node.Num(1))
        string = 'a=c=1\n'
        matcher = GetDynamicMatcher(node)
        matched_string = matcher.GetSource()
        self.assertNotEqual(string, matched_string)


    def testMatchMultiAssignWithWS(self):
        node = create_node.Assign(['a', 'b'], create_node.Num(1))
        string = 'a\t=\t     b \t  =1 \t'
        self._assert_matched_source(node, string)

    def testMatchMultiAssignWithWSAndComment(self):
        node = create_node.Assign(['a', 'b'], create_node.Num(1))
        string = 'a\t=\t     b \t  =1 \t #comment'
        self._assert_matched_source(node, string)

    @pytest.mark.xfail(strict=True)
    def testMatchMultiAssignNameWithWSAndComment(self):
        node = create_node.Assign(['a', 'b'], create_node.Name('c'))
        string = 'a\t=\t     b \t  =c \t #comment'
        self._assert_matched_source(node, string)

    def testNotMatchMultiAssignWithWS(self):
        node = create_node.Assign(['a', 'b'], create_node.Num(1))
        string = 'a\t=\t     bb \t  =1 \t'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(source_match.BadlySpecifiedTemplateError):
            matcher.Match(string)


    def _assert_matched_source(self, node, string):
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)
