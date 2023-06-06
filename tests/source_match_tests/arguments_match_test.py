import unittest

import pytest

from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError

from fun_with_ast.manipulate_node import create_node


class ArgumentsMatcherTest(unittest.TestCase):

    def testEmpty(self):
        node = create_node.arguments()
        string = ''
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)


    def testSingleArg(self):
        node = create_node.arguments(args=['a'])
        string = 'a'
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testMultipleArgs(self):
        node = create_node.arguments(args=['a', 'b'])
        string = 'a, b'
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testDefault(self):
        node = create_node.arguments(args=['a'], defaults=['b'])
        string = 'a=b'
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testDefaults(self):
        node = create_node.arguments(args=['a', 'c'], defaults=['b', 'd'])
        string = 'a=b, c=d'
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testArgsAndDefaults(self):
        node = create_node.arguments(
            args=['e', 'f', 'a', 'c'], defaults=['b', 'd'])
        string = 'e, f, a=b, c=d'
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testArgsDefaultsVarargs(self):
        node = create_node.arguments(
            args=['e', 'f', 'a', 'c'], defaults=['b', 'd'],
            vararg='args')
        string = 'e, f, a=b, c=d, *args'
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testMatchArgsDefaultsBool(self):
        node = create_node.arguments(
            args=['a'], defaults=[False])
        string = 'a = False'
#        matcher = GetDynamicMatcher(node)
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testMatchArgsDefaultsConst(self):
        node = create_node.arguments(
            args=['a'], defaults=[1])
        string = 'a = 1 \t  '
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def testNoMatchArgsDefaultsConst(self):
        node = create_node.arguments(
            args=['a'], defaults=[2])
        string = 'a = 1 \t  '
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher._match(string)

    def testArgsDefaultsVarargsKwargs(self):
        node = create_node.arguments(
            args=['e', 'f', 'a', 'c'], defaults=['b', 'd'],
            vararg='args', kwarg='kwargs')
        string = 'e, f, a=b, c=d, *args, **kwargs'
        matcher = GetDynamicMatcher(node)
        self._validate_match(matcher, string)

    def _validate_match(self, matcher, string):
        matcher._match(string)
        matched_text = matcher.GetSource()
        self.assertEqual(string, matched_text)
