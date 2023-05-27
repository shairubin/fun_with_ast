import unittest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.dynamic_matcher import GetDynamicMatcher


class TryExceptMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Try(
            [create_node.Pass()],
            [create_node.ExceptHandler(None, None, [create_node.Pass()])])

        string = """try:\n\tpass\nexcept:\n\tpass\n"""
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())


    def testMatchMultipleExceptHandlers(self):
        node = create_node.Try(
            [create_node.Expr(create_node.Name('a'))],
            [create_node.ExceptHandler('TestA'),
             create_node.ExceptHandler('TestB')])
        string = """try:
  a 
except TestA:
  pass
except TestB:
  pass
"""
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchExceptAndOrElse(self):
        node = create_node.Try(
            [create_node.Expr(create_node.Name('a'))],
            [create_node.ExceptHandler()],
            orelse=[create_node.Pass()])
        string = """try:
  a
except:
  pass
else:
  pass
"""
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchWithEmptyLine(self):
        node = create_node.Try(
            [create_node.Expr(create_node.Name('a'))],
            [create_node.ExceptHandler('Exception1', 'e')])
        string = """try:
  a

except Exception1 as e:

  pass
"""
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
