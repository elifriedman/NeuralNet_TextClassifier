import theano.tensor as T
import numpy as np
import theano

class GradObj:
  def __init__(self,saves):
    self.saves = saves
    self.len = len(saves)

  def fn(self,inps,theanofn):
    res = theanofn(*(inps+self.saves))
    self.saves = res[-self.len:]
    return res[:len(res) - self.len]

def SGD(params, grad, learning_rate,inputs=[],outputs=[],updates=[],givens=[]):
  updates = [(param, param - learning_rate*gparam) for param, gparam in zip(params,grad)]
  if "grad" in outputs:
    outputs.remove("grad")
    outputs.extend(grad)
  return theano.function(
            inputs = inputs,
            outputs = outputs,
            updates = updates,
            givens = givens
        )

def SGD_momentum(params, grad, learning_rate, momentum_rate,inputs=[],outputs=[],updates=[],givens=[]):
  prev_deltas = [T.matrix('prev_delta').reshape(gparam.shape) for gparam in grad]
  deltas = [-learning_rate*gparam + momentum_rate*prev_delta
            for gparam, prev_delta in zip(grad, prev_deltas)]
  update = [(param, param + delta) for param, delta in zip(params, deltas)]
  if "grad" in outputs:
    outputs.remove("grad")
    outputs.extend(grad)
  theanofn = theano.function(
            inputs = inputs + prev_deltas,
            outputs = outputs + deltas,
            updates = update,
            givens = givens
        )
  g = GradObj([np.zeros_like(p.get_value()) for p in params])
  return lambda inps: g.fn(inps,theanofn)

def ada_grad(params,grad,learning_rate,inputs=[],outputs=[],updates=[],givens=[]):
  fudge_factor = 1E-6
  running_grad = [T.matrix('running_grad').reshape(gparam.shape) for gparam in grad]
  new_grad = [g**2 + rg for g,rg in zip(grad,running_grad)]
  update = [(p,p - learning_rate*gp/(fudge_factor + T.sqrt(ng))) for ng,p,gp in zip(new_grad,params,grad)]
  if "grad" in outputs:
    outputs.remove("grad")
    outputs.extend(grad)
  theanofn = theano.function(
            inputs = inputs + running_grad,
            outputs = outputs + new_grad,
            updates = update,
            givens = givens
        )
  g = GradObj([np.zeros_like(p.get_value()) for p in params])
  return lambda inps: g.fn(inps,theanofn)


"""
The MIT License (MIT)
Copyright (c) 2015 Alec Radford
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Adam code based on code from: https://gist.github.com/Newmu/acb738767acb4788bac3
Adam updates based on http://arxiv.org/abs/1412.6980v8
"""    

def adam(params,grads, lr=0.0002, b1=0.1, b2=0.001, e=1e-8,inputs=[],outputs=[],givens=[]):
    updates = []
    i = theano.shared(np.asarray(0.,dtype=theano.config.floatX))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return theano.function(inputs=inputs,
                           outputs=outputs,
                           givens=givens,
                           updates=updates)