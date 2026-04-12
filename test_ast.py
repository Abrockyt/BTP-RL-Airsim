import ast
class V(ast.NodeVisitor):
    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Tuple):
                for e in t.elts:
                    if isinstance(e, ast.Name) and getattr(e, 'id', '') == 'state_vec':
                        print(f"Assign state_vec at line {node.lineno}")
        self.generic_visit(node)
    def visit_Name(self, node):
        if node.id == 'state_vec' and isinstance(node.ctx, ast.Load):
            print(f"Load state_vec at line {node.lineno}")
        self.generic_visit(node)
V().visit(ast.parse(open('iot_projet_gui.py', encoding='utf-8').read()))
