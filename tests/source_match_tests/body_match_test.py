import unittest

import create_node
import source_match
from dynamic_matcher import GetDynamicMatcher


class BodyPlaceholderTest(unittest.TestCase):

    def testMatchSimpleField(self):
        body_node = create_node.Expr(create_node.Name('foobar'))
        node = create_node.Module(body_node)
        placeholder = source_match.BodyPlaceholder('body')
        matched_text = placeholder.Match(node, 'foobar\n')
        self.assertEqual(matched_text, 'foobar\n')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar\n')

    def testMatchFieldAddsEmptySyntaxFreeLine(self):
        body_node_foobar = create_node.Expr(create_node.Name('foobar'))
        body_node_a = create_node.Expr(create_node.Name('a'))
        node = create_node.Module(body_node_foobar, body_node_a)
        placeholder = source_match.BodyPlaceholder('body')
        matched_text = placeholder.Match(node, 'foobar\n\na\n')
        self.assertEqual(matched_text, 'foobar\n\na\n')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar\n\na\n')

    def testMatchFieldAddsEmptySyntaxFreeLineWithComment(self):
        body_node_foobar = create_node.Expr(create_node.Name('foobar'))
        body_node_a = create_node.Expr(create_node.Name('a'))
        node = create_node.Module(body_node_foobar, body_node_a)
        placeholder = source_match.BodyPlaceholder('body')
        matched_text = placeholder.Match(node, 'foobar\n#blah\na\n')
        self.assertEqual(matched_text, 'foobar\n#blah\na\n')
        test_output = placeholder.GetSource(node)
        self.assertEqual(test_output, 'foobar\n#blah\na\n')


    def testMatchPass(self):
        body_node_pass = create_node.Pass()
        node = create_node.Module(body_node_pass)
        placeholder = source_match.BodyPlaceholder('body')
        matched_text = placeholder.Match(node, 'pass')
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
        matched_text = matcher.Match(text_to_match)
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
        matched_text = matcher.Match(text_to_match)
        expected_match = """def a():
  foobar #blah
  a
"""
        self.assertEqual(matched_text, expected_match)
