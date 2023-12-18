import ast


class ConstantForJstr(ast.Constant):
    def __init__(self, value):
        super(ConstantForJstr, self).__init__(value)
