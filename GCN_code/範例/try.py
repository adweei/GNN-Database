import torch as th
import dgl

g = dgl.DGLGraph()
g.add_nodes(3)
g.ndata['x'] = th.tensor([[1.], [2.], [3.]])
g.add_edges([0, 1], [1, 2])
# Define the function for sending node features as messages.
def send_source(edges): return {'m': edges.src['x']}
# Set the function defined to be the default message function.
g.register_message_func(send_source)
# Sum the messages received and use this to replace the original node feature.
def simple_reduce(nodes): return {'x': nodes.mailbox['m'].sum(1)}
# Set the function defined to be the default message reduce function.
g.register_reduce_func(simple_reduce)