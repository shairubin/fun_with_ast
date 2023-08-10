import unittest
import ast
def ExpandTree(node):
    node_fields = []
    to_expand = [node]
    while to_expand:
        current = to_expand.pop()
        if isinstance(current, ast.AST):
            node_fields.append(current.__class__)
            for field_name, child in ast.iter_fields(current):
                node_fields.append(field_name)
                if isinstance(child, (list, tuple)):
                    for item in child:
                        to_expand.append(item)
                else:
                    a = to_expand.append(child)
        else:
            node_fields.append(current)
    print("\n", node_fields, "\n")
    return node_fields


class CreateNodeTestBase(unittest.TestCase):

    def assertNodesEqual(self, node1, node2, equal=True):
        node1_fields = ExpandTree(node1)
        node2_fields = ExpandTree(node2)
        if equal:
            self.assertEqual(node1_fields, node2_fields)
        else:
            self.assertNotEqual(node1_fields, node2_fields)