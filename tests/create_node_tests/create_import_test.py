import create_node
from fun_with_ast.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateImportTest(CreateNodeTestBase):

    def testImport(self):
        expected_string = """import foo"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Import(import_part='foo')
        self.assertNodesEqual(expected_node, test_node)

    def testImportAs(self):
        expected_string = """import foo as foobar"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Import(import_part='foo', asname='foobar')
        self.assertNodesEqual(expected_node, test_node)

    def testImportFrom(self):
        expected_string = """from bar import foo"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Import(import_part='foo', from_part='bar')
        self.assertNodesEqual(expected_node, test_node)

    def testImportFromAs(self):
        expected_string = """from bar import foo as foobar"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.Import(
            import_part='foo', from_part='bar', asname='foobar')
        self.assertNodesEqual(expected_node, test_node)
