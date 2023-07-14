from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateClassDefTest(CreateNodeTestBase):

    def testBasicClass(self):
        expected_string = 'class TestClass():\n  pass'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.ClassDef('TestClass', body=[create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)

    def testBases(self):
        expected_string = 'class TestClass(Base1, Base2):\n  pass'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.ClassDef(
            'TestClass', bases=['Base1', 'Base2'], body=[create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)

    def testBody(self):
        expected_string = 'class TestClass(Base1, Base2):\n  a'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.ClassDef(
            'TestClass', bases=['Base1', 'Base2'],
            body=[create_node.Expr(create_node.Name('a'))])
        self.assertNodesEqual(expected_node, test_node)

    def testDecoratorList(self):
        expected_string = '@dec\n@dec2()\nclass TestClass():\n  pass\n'
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.ClassDef(
            'TestClass', body=[create_node.Pass()],
            decorator_list=[create_node.Name('dec'),
                            create_node.Call('dec2')])
        self.assertNodesEqual(expected_node, test_node)
