import numpy as np


def normalize(m):
    return m / m.sum()


def forward(t_m, o_m, fw):
    # print("t_model\n", t_m, "\ns_model\n", o_m, "\nfw\n", fw)
    res = o_m * np.transpose(t_m) * fw
    return normalize(res)


def backward(t_m, o_m, bw):
    # print("t_model\n", t_m, "\ns_model\n", o_m, "\nbw\n", bw)
    return t_m * o_m * bw


def forward_backward(ev, prior, T, O_t, O_f):
    t = len(ev)
    # forward messages
    fw = [np.matrix([0.0, 0.0]) for _ in range(t + 1)]
    # backward messages
    bw = np.matrix([[1.0], [1.0]])
    # smoothed messages
    sv = [np.matrix([0.0, 0.0]) for _ in range(t)]

    # first element in fw is set to prior
    fw[0] = prior

    for i in range(t):
        # choosing true or false values depending on the current evidence
        o_m = O_t if ev[i] else O_f

        fw[i + 1] = forward(T, o_m, fw[i])
        print("f(%s:%s) = %s" % (1, i+1, fw[i+1]))

    for i in reversed(range(0, t)):
        # sv[i] is the normalized vector of the multiplication of calculated forward messages, and the backward messages
        sv[i] = normalize(np.multiply(fw[i+1], bw))
        # choosing true or false values depending on the current evidence
        o_m = O_t if ev[i] else O_f
        # updating the backward messages for new iterations
        bw = backward(T, o_m, bw)
    return sv


def exercise_2(ev):
    # setting the example specific values
    T = np.matrix([[0.7, 0.3], [0.3, 0.7]])
    O_t = np.matrix([[0.9, 0.0], [0.0, 0.2]])
    O_f = np.matrix([[0.1, 0.0], [0.0, 0.8]])
    prior = np.matrix([[0.5], [0.5]])
    print("Running exercise with evidence:", ev)
    sv = forward_backward(ev, prior, T, O_t, O_f)
    for i in range(len(sv)):
        print("P(X_%s|e(1:%s))= %s" % (i + 1, len(sv), sv[i]))


exercise_2([True, True])
exercise_2([True, True, False, True, True])