from manipulate_node import create_node
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateSyntaxFreeLineTest(CreateNodeTestBase):

    def testEmpty(self):
        expected_string = ''
        test_node = create_node.SyntaxFreeLine(comment=None)
        self.assertEqual(expected_string, test_node.full_line)

    def testSimpleComment(self):
        expected_string = '#Comment'
        test_node = create_node.SyntaxFreeLine(
            comment='Comment', col_offset=0, comment_indent=0)
        self.assertEqual(expected_string, test_node.full_line)

    def testColOffset(self):
        expected_string = '  #Comment'
        test_node = create_node.SyntaxFreeLine(
            comment='Comment', col_offset=2, comment_indent=0)
        self.assertEqual(expected_string, test_node.full_line)

    def testCommentIndent(self):
        expected_string = '  # Comment'
        test_node = create_node.SyntaxFreeLine(
            comment='Comment', col_offset=2, comment_indent=1)
        self.assertEqual(expected_string, test_node.full_line)

    def testSetFromSrcLineEmpty(self):
        test_input = '\n'
        test_node = create_node.SyntaxFreeLine()
        test_node.SetFromSrcLine(test_input)
        self.assertEqual(test_node.col_offset, 0)
        self.assertEqual(test_node.comment_indent, 0)
        self.assertEqual(test_node.comment, None)

    def testSetFromSrcLineVeryShortComment(self):
        test_input = '#\n'
        test_node = create_node.SyntaxFreeLine()
        test_node.SetFromSrcLine(test_input)
        self.assertEqual(test_node.col_offset, 0)
        self.assertEqual(test_node.comment_indent, 0)
        self.assertEqual(test_node.comment, '')

    def testSetFromSrcLineComment(self):
        test_input = '#Comment\n'
        test_node = create_node.SyntaxFreeLine()
        test_node.SetFromSrcLine(test_input)
        self.assertEqual(test_node.col_offset, 0)
        self.assertEqual(test_node.comment_indent, 0)
        self.assertEqual(test_node.comment, 'Comment')

    def testSetFromSrcLineIndentedComment(self):
        test_input = '  #Comment\n'
        test_node = create_node.SyntaxFreeLine()
        test_node.SetFromSrcLine(test_input)
        self.assertEqual(test_node.col_offset, 2)
        self.assertEqual(test_node.comment_indent, 0)
        self.assertEqual(test_node.comment, 'Comment')

    def testSetFromSrcLineOffsetComment(self):
        test_input = '  # Comment\n'
        test_node = create_node.SyntaxFreeLine()
        test_node.SetFromSrcLine(test_input)
        self.assertEqual(test_node.col_offset, 2)
        self.assertEqual(test_node.comment_indent, 1)
        self.assertEqual(test_node.comment, 'Comment')

    def testSetFromSrcLineDoubleComment(self):
        test_input = '  # Comment # More comment\n'
        test_node = create_node.SyntaxFreeLine()
        test_node.SetFromSrcLine(test_input)
        self.assertEqual(test_node.col_offset, 2)
        self.assertEqual(test_node.comment_indent, 1)
        self.assertEqual(test_node.comment, 'Comment # More comment')

    def testSetFromSrcLineNoComment(self):
        test_input = '  Comment\n'
        test_node = create_node.SyntaxFreeLine()
        with self.assertRaises(ValueError):
            test_node.SetFromSrcLine(test_input)
