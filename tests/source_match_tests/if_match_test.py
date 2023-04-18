import unittest

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
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicIfElse2(self):
        node = create_node.If(
            create_node.Name('True'), body=[create_node.Pass()], orelse=[create_node.Pass()])
        string = """if True:\n  pass\nelse: \t\n  pass"""
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

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
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

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
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


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
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

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
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

