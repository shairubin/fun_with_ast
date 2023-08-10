import ast


class ResetMatch():
    def __init__(self, node):
        self.node = node
        self._validate_node()
        
    def reset_match(self):
        nodes = [node for node in ast.walk(self.node)]
        for node in nodes:
            if not hasattr(node, 'node_matcher') and not isinstance(node, ast.Load) and not isinstance(node, ast.Store):
                raise Exception('node does not have node_matcher attribute')
            elif isinstance(node, ast.Load) or isinstance(node, ast.Store):
                continue
            else:
                node.node_matcher.matched = False
                node.node_matcher.matched_source = None

    def _validate_node(self):
        nodes = [node for node in ast.walk(self.node)]
        for node in nodes:
            if not hasattr(node, 'node_matcher') and not isinstance(node, ast.Load) and not isinstance(node, ast.Store):
                raise Exception('node does not have node_matcher attribute')
            elif isinstance(node, ast.Load) or isinstance(node, ast.Store):
                continue
            elif hasattr(node.node_matcher , 'matched_text'):
                raise ValueError()
            elif node.node_matcher.matched and not node.node_matcher.matched_source:
                if not node.node_matcher.matched_source == '':
                    raise ValueError("one of the fields is not set")
            elif not node.node_matcher.matched and node.node_matcher.matched_source:
                raise ValueError("one of the fields is not set")