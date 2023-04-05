import unittest

import pytest

import create_node
import source_match


class ArgumentsMatcherTest(unittest.TestCase):

    def testEmpty(self):
        node = create_node.arguments()
        string = ''
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testSingleArg(self):
        node = create_node.arguments(args=['a'])
        string = 'a'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMultipleArgs(self):
        node = create_node.arguments(args=['a', 'b'])
        string = 'a, b'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testDefault(self):
        node = create_node.arguments(args=['a'], defaults=['b'])
        string = 'a=b'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testDefaults(self):
        node = create_node.arguments(args=['a', 'c'], defaults=['b', 'd'])
        string = 'a=b, c=d'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testArgsAndDefaults(self):
        node = create_node.arguments(
            args=['e', 'f', 'a', 'c'], defaults=['b', 'd'])
        string = 'e, f, a=b, c=d'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        matched_source = matcher.GetSource()
        self.assertEqual(string, matched_source)

    def testArgsDefaultsVarargs(self):
        node = create_node.arguments(
            args=['e', 'f', 'a', 'c'], defaults=['b', 'd'],
            vararg='args')
        string = 'e, f, a=b, c=d, *args'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchArgsDefaultsBool(self):
        node = create_node.arguments(
            args=['a'], defaults=[False])
        string = 'a = False'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchArgsDefaultsConst(self):
        node = create_node.arguments(
            args=['a'], defaults=[1])
        string = 'a = 1 \t  '
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testNioMatchArgsDefaultsConst(self):
        node = create_node.arguments(
            args=['a'], defaults=[2])
        string = 'a = 1 \t  '
        matcher = source_match.GetMatcher(node)
        with pytest.raises(source_match.BadlySpecifiedTemplateError):
            matcher.Match(string)

    def testArgsDefaultsVarargsKwargs(self):
        node = create_node.arguments(
            args=['e', 'f', 'a', 'c'], defaults=['b', 'd'],
            vararg='args', kwarg='kwargs')
        string = 'e, f, a=b, c=d, *args, **kwargs'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
