"""Model-free scalar controls for encoder.py DecisionHead, revision
02b55d7076f15129e59ab3f94783f32c4b088674. Python stdlib only, no tensors/imports
from reference code. This is elementary operation parity, not checkpoint parity.

Tiny fixed weights: type rows [+0.002*t,-0.002*t,0...]; Q/K/V diagonal
0.125/0.25/0.5; output diagonal0.125; FFN first64 diagonal0.5 with alternating
bias+/-0.02, remaining192 rows zero; second FFN diagonal0.25 on first64;
all norms unit/bias0 eps1e-5; scorer identity, exact GELU, [1,-1,0...] bias0.25.
"""
import hashlib
import json
import math
from pathlib import Path

D = 64

def norm(x):
    mean = sum(x) / D
    variance = sum((v - mean) ** 2 for v in x) / D
    return [(v - mean) / math.sqrt(variance + 1e-5) for v in x]

def gelu(x):
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def forward(hidden, lengths, types, markers, ignore_mask=False, causal=False):
    result = []
    for batch, original in enumerate(hidden):
        h = [row.copy() for row in original]
        for row in h:
            row[0] += 0.002 * types[batch]
            row[1] -= 0.002 * types[batch]
        normalized = [norm(row) for row in h]
        q = [[x * 0.125 for x in row] for row in normalized]
        k = [[x * 0.25 for x in row] for row in normalized]
        v = [[x * 0.5 for x in row] for row in normalized]
        attended = []
        for position, query in enumerate(q):
            count = len(h) if ignore_mask else lengths[batch]
            if causal:
                count = min(count, position + 1)
            scores = [sum(x * y for x, y in zip(query, key)) * 0.125 for key in k[:count]]
            exp = [math.exp(score - max(scores)) for score in scores]
            probabilities = [value / sum(exp) for value in exp]
            attended.append([h[position][d] + 0.125 * sum(probabilities[j] * v[j][d] for j in range(count)) for d in range(D)])
        encoded = []
        for row in attended:
            normalized = norm(row)
            feedforward = [max(0.0, x * 0.5 + (0.02 if d % 2 == 0 else -0.02)) * 0.25 for d, x in enumerate(normalized)]
            encoded.append([x + y for x, y in zip(row, feedforward)])
        answers = []
        for position in markers[batch]:
            selected = norm(encoded[position])
            answers.append(gelu(selected[0]) - gelu(selected[1]) + 0.25)
        result.append(answers)
    return result

hidden = []
for b in range(2):
    rows = []
    for p in range(3):
        rows.append([((b * 7 + p * 11 + d * 3) % 23 - 11) / 128.0 for d in range(D)])
    hidden.append(rows)
# A padded row with a deliberately different direction makes ignored padding visible.
hidden[1][2] = [2.0 if d == 0 else -1.0 if d == 1 else 0.0 for d in range(D)]
lengths, types, markers = [3, 2], [1, 2], [[2, 0], [1]]
fixture = dict(hidden=hidden, lengths=lengths, types=types, markers=markers,
               expected=forward(hidden, lengths, types, markers),
               ignored_mask=forward(hidden, lengths, types, markers, ignore_mask=True),
               wrong_causal=forward(hidden, lengths, types, markers, causal=True),
               exact_gelu_one=gelu(1.0),
               tanh_gelu_one=0.5 * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (1.0 + 0.044715))))
path = Path(__file__).with_name("head_scalar.json")
content = json.dumps(fixture, indent=2) + "\n"
path.write_text(content)
print("FIXTURE=" + str(path))
print("SHA256=" + hashlib.sha256(content.encode()).hexdigest())
print("EXPECTED=" + json.dumps(fixture["expected"]))
print("IGNORED_MASK=" + json.dumps(fixture["ignored_mask"]))
print("WRONG_CAUSAL=" + json.dumps(fixture["wrong_causal"]))
print("EXACT_GELU_ONE=" + repr(fixture["exact_gelu_one"]))
print("TANH_GELU_ONE=" + repr(fixture["tanh_gelu_one"]))
