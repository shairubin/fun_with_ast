import unittest

import pytest

from fun_with_ast.dynamic_matcher import GetDynamicMatcher
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError

from manipulate_node import create_node


class ArgumentsMatcherTest(unittest.TestCase):

    def testEmpty(self):
        node = create_node.arguments()
        string = ''
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testSingleArg(self):
        node = create_node.arguments(args=['a'])
        string = 'a'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMultipleArgs(self):
        node = create_node.arguments(args=['a', 'b'])
        string = 'a, b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testDefault(self):
        node = create_node.arguments(args=['a'], defaults=['b'])
        string = 'a=b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testDefaults(self):
        node = create_node.arguments(args=['a', 'c'], defaults=['b', 'd'])
        string = 'a=b, c=d'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testArgsAndDefaults(self):
        node = create_node.arguments(
            args=['e', 'f', 'a', 'c'], defaults=['b', 'd'])
        string = 'e, f, a=b, c=d'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_source = matcher.GetSource()
        self.assertEqual(string, matched_source)

    def testArgsDefaultsVarargs(self):
        node = create_node.arguments(
            args=['e', 'f', 'a', 'c'], defaults=['b', 'd'],
            vararg='args')
        string = 'e, f, a=b, c=d, *args'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchArgsDefaultsBool(self):
        node = create_node.arguments(
            args=['a'], defaults=[False])
        string = 'a = False'
#        matcher = GetDynamicMatcher(node)
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchArgsDefaultsConst(self):
        node = create_node.arguments(
            args=['a'], defaults=[1])
        string = 'a = 1 \t  '
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testNoMatchArgsDefaultsConst(self):
        node = create_node.arguments(
            args=['a'], defaults=[2])
        string = 'a = 1 \t  '
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.Match(string)

    def testArgsDefaultsVarargsKwargs(self):
        node = create_node.arguments(
            args=['e', 'f', 'a', 'c'], defaults=['b', 'd'],
            vararg='args', kwarg='kwargs')
        string = 'e, f, a=b, c=d, *args, **kwargs'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
