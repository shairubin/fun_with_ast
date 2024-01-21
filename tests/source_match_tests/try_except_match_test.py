from fun_with_ast.manipulate_node import create_node
from tests.source_match_tests.base_test_utils import BaseTestUtils


class TryExceptMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Try(
            [create_node.Pass()],
            [create_node.ExceptHandler(None, None, [create_node.Pass()])])

        string = """try:\n\tpass\nexcept:\n\tpass\n"""
        self._assert_match(node, string)

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
        self._assert_match(node, string)

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
        self._assert_match(node, string)

    def _assert_match(self, node, string):
        #matcher = GetDynamicMatcher(node)
        #matcher.do_match(string)
        #self.assertEqual(string, matcher.GetSource())
        self._verify_match(node, string)

    def testMatchWithEmptyLine(self):
        node = create_node.Try(
            [create_node.Expr(create_node.Name('a'))],
            [create_node.ExceptHandler('Exception1', 'e')])
        string = """try:
  a

except Exception1 as e:

  pass
"""
        self._assert_match(node, string)
