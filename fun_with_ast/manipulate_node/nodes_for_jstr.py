import ast


class ConstantForJstr(ast.Constant):
    def __init__(self, value):
        super(ConstantForJstr, self).__init__(value)

class NameForJstr(ast.Name):
    def __init__(self, id, ctx):
        super(NameForJstr, self).__init__(id, ctx)
