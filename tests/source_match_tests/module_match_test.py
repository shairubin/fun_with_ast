import unittest

import create_node
import source_match
from dynamic_matcher import GetDynamicMatcher


class ModuleMatcherTest(unittest.TestCase):
    def testModuleBasicFailed(self):
        node = create_node.Module(create_node.FunctionDef(name='myfunc', body=[
            create_node.AugAssign('a', create_node.Add(), create_node.Name('c'))]))
        string = 'def myfunc():\n \t a += c\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testModuleBasic(self):
        node = create_node.Module(create_node.FunctionDef(name='myfunc', body=[
            create_node.AugAssign('a', create_node.Add(), create_node.Name('c'))]))
        string = 'def myfunc():\n\ta += c\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)

    def testBasicMatch(self):
        node = create_node.Module(create_node.Expr(create_node.Name('a')))
        string = 'a\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchEndsWithComent(self):
        node = create_node.Module(create_node.Expr(create_node.Name('a')))
        string = '   a  \t  \n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithEmptyLines(self):
        node = create_node.Module(
            create_node.Expr(create_node.Name('a')),
            create_node.Expr(create_node.Name('b')))
        string = 'a\n\nb\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicMatchWithCommentLines(self):
        node = create_node.Module(
            create_node.Expr(create_node.Name('a')),
            create_node.Expr(create_node.Name('b')))
        string = 'a\n#blah\nb\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
