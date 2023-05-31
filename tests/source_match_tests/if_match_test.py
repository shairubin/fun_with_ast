import unittest

from fun_with_ast.get_source import GetSource
from fun_with_ast.manipulate_node.create_node import SyntaxFreeLine, GetNodeFromInput

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher


class IfMatcherTest(unittest.TestCase):
    def testSimpleIfElse(self):
        node = create_node.If(conditional=True, body=[create_node.Pass()], orelse=[create_node.Pass()])
        string = 'if       True:   \n   pass    \nelse:\n   pass \n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)
    def testSimpleIfElse2(self):
        node = create_node.If(conditional=create_node.Compare(create_node.Name('a'),'==', create_node.Num(2)),
                              body=[create_node.Pass()], orelse=[create_node.Pass()])
        string = 'if       a==2:   \n   pass    \nelse:\n   pass \n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)

    def testSimpleIfElse2WithComment(self):
        node = create_node.If(conditional=create_node.Compare(create_node.Name('a'),'==', create_node.Num(2)),
                              body=[create_node.Pass()], orelse=[create_node.Pass()])
        string = 'if       a==2:#comment   \n   pass    \nelse:\n   pass \n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)

    def testSimpleIfElseWithCommentAndSpeacses(self):
        node = create_node.If(conditional=True, body=[create_node.Pass()], orelse=[create_node.Pass()])
        string = 'if       True: #comment  \n pass    \nelse: # comment    \n pass#comment\n'
        self._assert_match_to_source(node, string, match_get_source=False)

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
                            body=[create_node.Assign('a', 1)])
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
        self._assert_match_to_source(node, string, match_get_source=False)

    def testBasicIfElse3(self):
        node = create_node.If(
            create_node.Name('True'), body=[create_node.Assign('a', 1)], orelse=[create_node.Assign('a', 2)])
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
        self._assert_match_to_source(node, string, match_get_source=False)

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

    def _assert_match_to_source(self, node, string, lines_in_body=1, match_get_source=True):
        assume_elif = False
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)
        self.assertEqual(len(node.body),lines_in_body)
        if 'elif' in string:
            assume_elif = True
        if match_get_source:
            source = GetSource(node, assume_no_indent=True, assume_elif=assume_elif)
            self.assertEqual(string, source)

    def testIfFromSource(self):
        string = "if (a and b):\n     a = 1\nelse:\n    a=2"
        if_node = GetNodeFromInput(string)
        if_node_matcher = GetDynamicMatcher(if_node)
        if_node_matcher.Match(string)

    def testIfFromSource(self):
        string = "if not a:\n     a = 1\nelse:\n    a=2"
        if_node = GetNodeFromInput(string)
        if_node_matcher = GetDynamicMatcher(if_node)
        if_node_matcher.Match(string)
