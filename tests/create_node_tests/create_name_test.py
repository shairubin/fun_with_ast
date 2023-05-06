import pytest

import create_node
from fun_with_ast.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateNameTest(CreateNodeTestBase):

    def testBaseicName(self):
        expected_string = '_b_'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Name('_b_')
        self.assertNodesEqual(expected_node, test_node)

    def testIlligalName(self):
        with  pytest.raises(ValueError):
            create_node.Name('b!')
        with  pytest.raises(ValueError):
            create_node.Name('b\n')
        with  pytest.raises(ValueError):
            create_node.Name('9b\n')


    def testNameWithLoad(self):
        expected_string = 'b = a'
        expected_node = GetNodeFromInput(expected_string).value
        test_node = create_node.Name('a', ctx_type=create_node.CtxEnum.LOAD)
        self.assertNodesEqual(expected_node, test_node)

    def testNameWithStore(self):
        expected_string = 'a = b'
        expected_node = GetNodeFromInput(expected_string).targets[0]
        test_node = create_node.Name('a', ctx_type=create_node.CtxEnum.STORE)
        self.assertNodesEqual(expected_node, test_node)

    def testNameWithDel(self):
        expected_string = 'del a'
        expected_node = GetNodeFromInput(expected_string).targets[0]
        test_node = create_node.Name('a', ctx_type=create_node.CtxEnum.DEL)
        self.assertNodesEqual(expected_node, test_node)
