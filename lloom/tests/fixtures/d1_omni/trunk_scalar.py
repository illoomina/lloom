"""Model-free scalar oracle for pinned encoder.py, not a checkpoint runner.
stdlib only: no network, model, framework, subprocess, or import side effects.
D4/H2/KV1, two layers, nonzero matrices; finite-prefix and centered-conv rules.
"""
import json
import math
from pathlib import Path

D, H, HD, L = 4, 2, 2, 4
COS, SIN = [1., .6, 0., -.8], [0., .8, 1., .6]
LENGTHS, PREFIXES = [4, 3], [2, 1]


def matrix(rows, cols, seed):
    return [[((i * cols + j) * 7 + seed * 3) % 17 / 32 - .25
             for j in range(cols)] for i in range(rows)]


def linear(x, w):
    return [sum(a * b for a, b in zip(x, row)) for row in w]


def norm(x):
    scale = 1 / math.sqrt(sum(a*a for a in x) / len(x) + 1e-5)
    return [a * scale for a in x]


def add(x, y):
    return [a+b for a, b in zip(x, y)]


def ffn(x, seed):
    g, u = linear(norm(x), matrix(D, D, seed)), linear(norm(x), matrix(D, D, seed+1))
    return linear([a / (1 + math.exp(-a)) * b for a, b in zip(g, u)], matrix(D, D, seed+2))


def run(ignore_mask=False, causal=False, ignore_right=False, no_rope=False, capture=None):
    rows = [[[((b*L+t)*D+i)*5 % 19 / 8 - 9/8 for i in range(D)] for t in range(L)] for b in range(2)]
    result = []
    for b, hidden in enumerate(rows):
        projected = [linear([a * (t < LENGTHS[b]) for a in norm(x)], matrix(12, D, 1))
                     for t, x in enumerate(hidden)]
        bx = [[x[i] * x[2*D+i] for i in range(D)] for x in projected]
        mixed = []
        conv_residual = []
        w = matrix(D, 3, 2)
        for t in range(L):
            y = []
            for i in range(D):
                left = bx[t-1][i] if t else 0.
                right = bx[t+1][i] if t+1 < L else 0.
                keep = ignore_right or t != PREFIXES[b]-1
                y.append((left*w[i][0] + bx[t][i]*w[i][1] + right*keep*w[i][2]) * projected[t][D+i])
            residual = add(hidden[t], linear(y, matrix(D, D, 3)))
            conv_residual.append(residual)
            mixed.append(add(residual, ffn(residual, 4)))
        x = [norm(v) for v in mixed]
        q = [[norm(linear(v, matrix(D,D,7))[h*HD:(h+1)*HD]) for h in range(H)] for v in x]
        k = [norm(linear(v, matrix(HD,D,8))) for v in x]
        v = [linear(row, matrix(HD,D,9)) for row in x]
        if not no_rope:
            def rotate(row, t):
                a, z = row
                return [a*COS[t]-z*SIN[t], z*COS[t]+a*SIN[t]]
            q = [[rotate(head,t) for head in row] for t,row in enumerate(q)]
            k = [rotate(row,t) for t,row in enumerate(k)]
        if capture is not None:
            capture.setdefault("after_conv", []).append(mixed)
            capture.setdefault("conv_residual", []).append(conv_residual)
            capture.setdefault("rotated_q", []).append([sum(row, []) for row in q])
            capture.setdefault("rotated_k", []).append(k)
            capture.setdefault("values", []).append(v)
        out = []
        for t in range(L):
            joined = []
            for head in q[t]:
                scores = []
                for key in range(L):
                    allowed = key < LENGTHS[b] and (t >= PREFIXES[b] or key < PREFIXES[b])
                    score = sum(a*z for a,z in zip(head,k[key])) / math.sqrt(HD)
                    if not ignore_mask and not allowed: score += -1e9
                    if causal and key > t: score += -1e9
                    scores.append(score)
                largest = max(scores)
                exps = [math.exp(s-largest) for s in scores]
                probs = [e/sum(exps) for e in exps]
                joined.extend(sum(p*vv[i] for p,vv in zip(probs,v)) for i in range(HD))
            residual = add(mixed[t], linear(joined, matrix(D,D,10)))
            out.append(norm(add(residual, ffn(residual, 11))))
        result.append(out)
    return result


if __name__ == '__main__':
    result = {name: run(**args) for name,args in [
        ('expected', {}), ('ignored_mask', {'ignore_mask': True}),
        ('wrong_causal', {'causal': True}), ('ignored_right', {'ignore_right': True}),
        ('no_rope', {'no_rope': True})]}
    capture = {}
    run(capture=capture)
    result.update(capture)
    path = Path(__file__).with_suffix('.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(path)
    for name in ["expected", "ignored_mask", "wrong_causal", "ignored_right", "no_rope"]:
        rows = result[name]
        delta=max(abs(a-b) for ar,br in zip(result['expected'],rows) for ax,bx in zip(ar,br) for a,b in zip(ax,bx))
        print(f'{name}: max_delta={delta:.12g}')
