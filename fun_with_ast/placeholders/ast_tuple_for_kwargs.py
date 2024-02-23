import ast


class AstTupleForKwArgs(ast.Tuple):
    def __init__(self, source):
        self.ctx = source.ctx
        self.elts = source.elts
        self.dims = source.dims
        self.col_offset = source.col_offset
        self.lineno = source.lineno
        self.end_col_offset = source.end_col_offset
        self.end_lineno = source.end_lineno
