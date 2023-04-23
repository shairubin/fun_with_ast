import unittest

from fun_with_ast.create_node import SyntaxFreeLine

import create_node
import source_match
from dynamic_matcher import GetDynamicMatcher


class IfMatcherTest(unittest.TestCase):
    def testSimpleIfElse(self):
        node = create_node.If(conditional=True, body=[create_node.Pass()], orelse=[create_node.Pass()])
        string = 'if       True:   \n pass    \nelse:\n pass \n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)

    def testSimpleIfElseWithCommentAndSpeacses(self):
        node = create_node.If(conditional=True, body=[create_node.Pass()], orelse=[create_node.Pass()])
        string = 'if       True: #comment  \n pass    \nelse: # comment    \n pass#comment\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)

    def testSimpleIf(self):
        node = create_node.If(conditional=True, body=[create_node.Pass()])
        string = 'if       True:\n pass         '
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)

    def testBasicIf(self):
        node = create_node.If(
            create_node.Name('True'),
                            body=[create_node.Pass()])
        string = """if True:\n  pass"""
        self._assert_match_to_source(node, string)

    def testBasicIf2(self):
        node = create_node.If(
            create_node.Name('True'),
                            body=[create_node.Assign('a',1)])
        string = """if True:\n  a=1"""
        self._assert_match_to_source(node, string)

    def testBasicIfwithEmptyLine(self):
        node = create_node.If(
            create_node.Name('True'),
                            body=[create_node.Pass()])
        string = """if True:\n\n  pass"""
        self._assert_match_to_source(node, string, 2)
        assert node.body[0].full_line == ''
        assert isinstance(node.body[0], SyntaxFreeLine)

    def testBasicIfElse2(self):
        node = create_node.If(
            create_node.Name('True'), body=[create_node.Pass()], orelse=[create_node.Pass()])
        string = """if True:\n  pass\nelse: \t\n  pass"""
        self._assert_match_to_source(node, string)

    def testBasicIfElse3(self):
        node = create_node.If(
            create_node.Name('True'), body=[create_node.Assign('a',1)], orelse=[create_node.Assign('a',2)])
        string = """if True:\n  a=1\nelse:\n  a=2"""
        self._assert_match_to_source(node, string)

    def testBasicIfElif(self):
        node = create_node.If(
            create_node.Name('True'),
            body=[create_node.Pass()],
            orelse=[create_node.If(create_node.Name('False'), body=[create_node.Pass()])])
        string = """if True:
  pass
elif False:
  pass
"""
        self._assert_match_to_source(node, string)

    def testBasicIfElifwWithWSAndComment(self):
        node = create_node.If(
        create_node.Name('True'),
        body=[create_node.Pass()],
        orelse=[create_node.If(create_node.Name('False'), body=[create_node.Pass()])])
        string = """if True:    \t #comment
      pass  \t
    elif False:    \t # comment  
      pass \t
"""
        self._assert_match_to_source(node, string)

    def testIfInElse(self):
        node = create_node.If(
            create_node.Name('True'),
            body=[create_node.Pass()],
            orelse=[create_node.If(create_node.Name('False'),
                                   body=[create_node.Pass()])])
        string = """if True:
  pass
else:
  if False:
    pass
"""
        self._assert_match_to_source(node, string)

    def testIfAndOthersInElse(self):
        node = create_node.If(
            create_node.Name('True'), body=[create_node.Pass()],
            orelse=[create_node.If(create_node.Name('False'), body=[create_node.Pass()]),
                    create_node.Expr(create_node.Name('True'))])
        string = """if True:
  pass
else:
  if False:
    pass
  True
"""
        self._assert_match_to_source(node, string)

    def _assert_match_to_source(self, node, string, lines_in_body=1):
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)
        self.assertEqual(len(node.body),lines_in_body)
