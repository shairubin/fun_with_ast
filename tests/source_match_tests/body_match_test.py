import unittest

from  fun_with_ast.source_matchers.body import BodyPlaceholder
from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher


class BodyPlaceholderTest(unittest.TestCase):

    def testMatchSimpleField(self):
        body_node = create_node.Expr(create_node.Name('foobar'))
        node = create_node.Module(body_node)
        placeholder = BodyPlaceholder('body')
        matched_text = placeholder._match(node, 'foobar\n')
        self.assertEqual(matched_text, 'foobar\n')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar\n')

    def testMatchFieldAddsEmptySyntaxFreeLine(self):
        body_node_foobar = create_node.Expr(create_node.Name('foobar'))
        body_node_a = create_node.Expr(create_node.Name('a'))
        module_node = create_node.Module(body_node_foobar, body_node_a)
        placeholder = BodyPlaceholder('body')
        matched_text = placeholder._match(module_node, 'foobar\n\na\n')
        self.assertEqual(matched_text, 'foobar\n\na\n')
        test_output = placeholder.GetSource(module_node)
        self.assertEqual(test_output, 'foobar\n\na\n')

    def testMatchFieldAddsEmptySyntaxFreeLineWithComment(self):
        body_node_foobar = create_node.Expr(create_node.Name('foobar'))
        body_node_a = create_node.Expr(create_node.Name('a'))
        node = create_node.Module(body_node_foobar, body_node_a)
        placeholder = BodyPlaceholder('body')
        matched_text = placeholder._match(node, 'foobar\n#blah\na\n')
        self.assertEqual(matched_text, 'foobar\n#blah\na\n')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar\n#blah\na\n')


    def testMatchPass(self):
        body_node_pass = create_node.Pass()
        node = create_node.Module(body_node_pass)
        placeholder = BodyPlaceholder('body')
        matched_text = placeholder._match(node, 'pass')
        self.assertEqual(matched_text, 'pass')

    def testDoesntMatchAfterEndOfBody(self):
        body_node_foobar = create_node.Expr(create_node.Name('foobar'))
        body_node_a = create_node.Expr(create_node.Name('a'))
        node = create_node.FunctionDef('a', body=[body_node_foobar, body_node_a])
        matcher = GetDynamicMatcher(node)
        text_to_match = """def a():
  foobar
#blah
  a

# end comment
c
"""
        matched_text = matcher._match(text_to_match)
        expected_match = """def a():
  foobar
#blah
  a
"""
        self.assertEqual(matched_text, expected_match)

    def testDoesntMatchAfterEndOfBodyAndComments(self):
        body_node_foobar = create_node.Expr(create_node.Name('foobar'))
        body_node_a = create_node.Expr(create_node.Name('a'))
        node = create_node.FunctionDef('a', body=[body_node_foobar, body_node_a])
        matcher = GetDynamicMatcher(node)
        text_to_match = """def a():
  foobar #blah
  a

# end comment
c
"""
        matched_text = matcher._match(text_to_match)
        expected_match = """def a():
  foobar #blah
  a
"""
        self.assertEqual(matched_text, expected_match)
