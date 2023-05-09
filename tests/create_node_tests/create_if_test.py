from manipulate_node import create_node
from manipulate_node.create_node import GetNodeFromInput
from tests.create_node_tests.create_node_test_based import CreateNodeTestBase


class CreateIfTest(CreateNodeTestBase):

    def testBasicIf(self):
        expected_string = """if True:\n  pass"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.If(
            create_node.Constant(True),
            body=[create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)

    def testBasicIfElse(self):
        expected_string = """if True:\n  pass\nelse:\n  pass"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.If(conditional=create_node.Constant(True),
                                   body=[create_node.Pass()], orelse=[create_node.Pass()])
        self.assertNodesEqual(expected_node, test_node)

    def testBasicIfElif(self):
        expected_string = """if True:
  pass
elif False:
  pass
"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.If(
            create_node.Constant(True),
            body=[create_node.Pass()],
            orelse=[create_node.If(create_node.Constant(False), body=[create_node.Pass()])])
        self.assertNodesEqual(expected_node, test_node)

    def testIfInElse(self):
        expected_string = """if True:
  pass
else:
  if False:
    pass
"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.If(
            create_node.Constant(True), body=[create_node.Pass()],
            orelse=[create_node.If(conditional=create_node.Constant(False), body=[create_node.Pass()])])
        self.assertNodesEqual(expected_node, test_node)

    def testIfAndOthersInElse(self):
        expected_string = """if True:
  pass
else:
  if False:
    pass
  True
"""
        expected_node = GetNodeFromInput(expected_string)
        test_node = create_node.If(
            create_node.Constant(True), body=[create_node.Pass()],
            orelse=[create_node.If(conditional=create_node.Constant(False), body=[create_node.Pass()]),
                    create_node.Expr(create_node.Constant(True))])
        self.assertNodesEqual(expected_node, test_node)
