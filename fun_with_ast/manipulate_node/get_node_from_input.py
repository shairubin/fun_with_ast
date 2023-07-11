import ast


def GetNodeFromInput(string, body_index = 0, get_module = False):
    generator = FWANodeGenerator()
    return generator.GetNodeFromInputV1(string, body_index, get_module)

# will be used in next generation of the library
class FWANodeGenerator():
    def __init__(self):
        pass
    def GetNodeFromInputV1(self, string, body_index = 0, get_module = False):
        parse_result = ast.parse(string)
        if parse_result.body and get_module == False:
            node = parse_result.body[body_index]
        elif parse_result.body and get_module == True:
            node = parse_result
        else:
            return parse_result # empty Module
        if isinstance(node, ast.If) and 'elif' in string:
            node.is_elif = True if 'elif' in string else False
        node.default_quote = self._guess_default_quote_for_node(string)
        return node

    def _guess_default_quote_for_node(self, string):
        return "'"
