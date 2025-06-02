def eye_data(t, y, slots=100, tau=100):
    res = []
    for slot in range(slots - 2):
        X, Y = [], []
        for i in range(len(t)):
            x = t[i] // tau
            if slot <= x < slot + 3:
                X.append(t[i] - slot * tau)
                Y.append(y[i])
            elif x >= slot + 3:
                break
        res.append((X, Y))
    return res





