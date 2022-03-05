from ..spec import Type, Production
from ..dsl import Builder
from . import Node, HoleNode, AtomNode, ParamNode, ApplyNode

def derive_dfs(builder: Builder, node: Node, prod: Production):
	'''
	Expand the first HoleNode using the provided production by DFS. Returns a copy of the new program.
	'''
	# first find the first HoleNode
	if isinstance(node, HoleNode):
		# if this is the root node, still you need to check type
		if node.type != prod.lhs:
			raise Exception("Types don't match at root level, expect {}, got {}".format(node.type, prod.lhs))
		# if you reach here, then derive
		tmp_clist = [HoleNode(type=p) for p in prod.rhs]
		tmp_node = builder.make_node(prod, tmp_clist)
		return (True, tmp_node)
	elif isinstance(node, AtomNode):
		return (False, node)
	elif isinstance(node, ParamNode):
		return (False, node)
	elif isinstance(node, ApplyNode):
		# print("# [debug] node={}".format(node))
		# make copy of the node if already derived
		derived = False
		tmp_clist = []
		for i in range(len(node.children)):
			# print("# [debug] i={}".format(i))
			if derived:
				# if expanded, then just copy
				tmp_clist.append(node.children[i].make_copy())
			else:
				# not expanded, try to expand
				cderived, cnode = derive_dfs(builder, node.children[i], prod)
				if cderived:
					derived = True
					# you need to check the type
					if node.production.rhs[i] != cnode.production.lhs:
						raise Exception("Types don't match, expect {}, got {}".format(node.production.rhs[i], cnode.production.lhs))
					tmp_clist.append(cnode)
				else:
					# no hole in this branch and its children
					tmp_clist.append(node.children[i].make_copy())
		# print("# [debug] node.production={}, node={}, tmp_clist={}".format(node.production, node, tmp_clist))
		tmp_node = builder.make_node(node.production, tmp_clist)
		return (derived, tmp_node)
	else:
		raise NotImplementedError("Unsupported node type, got: {}".format(type(node)))

def get_hole_dfs(node: Node):
	'''
	Get the first HoleNode encountered by DFS and return it; if not found, i.e., the program is complete, return None.
	'''
	if isinstance(node, HoleNode):
		return node
	elif isinstance(node, AtomNode):
		return None
	elif isinstance(node, ParamNode):
		return None
	elif isinstance(node, ApplyNode):
		res = None
		for i in range(len(node.children)):
			if res is not None:
				# already found, break
				break
			res = get_hole_dfs(node.children[i])
		return res
	else:
		raise NotImplementedError("Unsupported node type, got: {}".format(type(node)))