import pytest

import create_node
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CommentCreateTest(CreateNodeTestBase):
    def testSimpleComment(self):
        expected_string = '#Comment\n'
        test_node = create_node.Comment('#Comment\n')
        self.assertEqual(expected_string, test_node.source_comment)

    def testSimpleCommentWithWS(self):
        expected_string = '#Comment\t'
        test_node = create_node.Comment('#Comment\t')
        self.assertEqual(expected_string, test_node.source_comment)

    def testErrorsComment(self):
        with pytest.raises(ValueError):
            create_node.Comment('Comment')
        with pytest.raises(ValueError):
            create_node.Comment(' #Comment')
        with pytest.raises(ValueError):
            create_node.Comment('')
