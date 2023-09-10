import unittest

import pytest
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput

from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class NameMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Name('foobar')
        string = 'foobar'
        self._verify_match(node, string)
    def testBasicMatchWiuthParans(self):
        node = create_node.Name('foobar')
        string = '(foobar)'
        self._verify_match(node, string)
    def testBasicMatchWiuthParansWithComment(self):
        node = create_node.Name('foobar')
        string = '(foobar) #comment'
        self._verify_match(node, string)

    def testBasicMatchWiuthParansWithCommentAndNL(self):
        node = create_node.Name('foobar')
        string = '(foobar): #comment'
#        self._verify_match(node, string)
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
           matcher.do_match(string)

    def testBasicMatchWiuthParans2(self):
        node = create_node.Name('foobar')
        string = ' \t (foobar \t ) \t '
        self._verify_match(node, string)

    def testBasicMatchWiuthParans2(self):
        node = create_node.Name('foobar')
        string = '(foobar))'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)



    def testBasicMatchWithWS(self):
        node = create_node.Name('foobar')
        string = ' \t  foobar \t'
        self._verify_match(node, string)
    def testBasicMatchWithWSAndComment(self):
        node = create_node.Name('foobar')
        string = ' \t  foobar \t #comment'
        self._verify_match(node, string)

    def testBasicMatchOnlyComment(self):
        node = create_node.Name('foobar')
        string = ' \t  foobar#comment'
        self._verify_match(node, string)

    def testBasicMatchOnlyComment(self):
        node = create_node.Name('foobar')
        string = ' \t #comment  foobar'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)

    def testIdChange(self):
        node = create_node.Name('foobar')
        string = 'foobar'
        self._verify_match(node, string)


    def testBasicMatch2(self):
        node = create_node.Name('a')
        string = 'a'
        self._verify_match(node, string)

    def testMatchWithWS(self):
        node = create_node.Name('a')
        string = 'a '
        self._verify_match(node, string)

    def testMatchWithComment(self):
        node = create_node.Name('a')
        string = 'a # comment'
        self._verify_match(node, string)

    def testNamesInModule(self):
        string = '(b)\n#\nn(a)\n'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testNamesInModule1(self):
        string = '(b)\n(a)\n'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testNamesInModule2(self):
        string = '(b)\n'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testNamesInModule3(self):
        string = 'b\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testLeadingSpaces(self):
        node = create_node.Name('a')
        string = '  a'
        self._verify_match(node, string)
        string = ' \t  a'
        self._verify_match(node, string)
        string = ' \t\n  a'
        matcher = GetDynamicMatcher(node)
        with self.assertRaises(BadlySpecifiedTemplateError):
            matcher.do_match(string)
