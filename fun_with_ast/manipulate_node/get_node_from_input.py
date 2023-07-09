import ast


def GetNodeFromInput(string, body_index = 0, get_module = False):
    parse_result = ast.parse(string)
    if parse_result.body and get_module == False:
        node = parse_result.body[body_index]
    elif parse_result.body and get_module == True:
        node = parse_result
    else:
        return parse_result # empty Module
    if isinstance(node, ast.If) and 'elif' in string:
        node.is_elif = True if 'elif' in string else False
    return node
