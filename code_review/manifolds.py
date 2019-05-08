
def deep_mlp(muNet, sigmaNet):
    """
    muNet is a dictionary containing the following keys:
        W0, W1, W2, b0, b1, b2
    sigmaNet is a dictionary containing the following keys:
        Wrbf, gammas, centers, zeta
    """
    manifold = {**muNet, **sigmaNet}
    manifold['dimension'] = muNet['W0'][1]
    return manifold
