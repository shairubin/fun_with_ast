class ManipulateIfNode():
    def __init__(self, node):
        self.node = node

    def add_nodes_to_body(self, nodes: list, location: int):
        if location > len(self.node.body):
            raise ValueError("location is out of range")
        if location < 0:
            raise ValueError("location must be positive")
        for node in nodes:
            self.node.body.insert(location, node)
        self._add_newlines()

    def _add_newlines(self):
        for node in self.node.body:
            node_source = node.matcher.GetSource()
            if node_source.endswith("\n"):
                continue
            node.matcher.add_newline_to_source()


