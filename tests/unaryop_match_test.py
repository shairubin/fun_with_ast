import unittest

import pytest

import create_node
import source_match


class UnaryOpMatcherTest(unittest.TestCase):

    def testUAddUnaryOp(self):
        node = create_node.UnaryOp(
            create_node.UAdd(),
            create_node.Name('a'))
        string = '+a'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testUSubUnaryOp(self):
        node = create_node.UnaryOp(
            create_node.USub(),
            create_node.Name('a'))
        string = '-a'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testNotUnaryOp(self):
        node = create_node.UnaryOp(
            create_node.Not(),
            create_node.Name('a'))
        string = 'not a'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testInvertUnaryOp(self):
        node = create_node.UnaryOp(
            create_node.Invert(),
            create_node.Name('a'))
        string = '~a'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testInvertUnaryOpWithWS(self):
        node = create_node.UnaryOp(
            create_node.Invert(),
            create_node.Name('a'))
        string = '~a    \t '
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    @pytest.mark.xfail(strict=True)
    def testInvertUnaryOpWithWSAndComment(self):
        node = create_node.UnaryOp(
            create_node.Invert(),
            create_node.Name('a'))
        string = '~a    \t #comment'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
