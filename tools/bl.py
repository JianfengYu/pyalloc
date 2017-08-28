import numpy as np
from scipy import linalg

class DefaultPara:
    # Risk aversion of the market
    delta = 2.5
    tau = 0.05

    delta_idz = 3.07
    tau_idz = 0.025

def blacklitterman(delta, weq, sigma, tau, P, Q, Omega) -> tuple:
    """
    This function performs the Black-Litterman blending of the prior and views into a new posterior estimate of the
    returns as described in the paper by He and Litterman.
    See "The Intuition Behind Black-Litterman Model  Portfolios", by He and Litterman, 1999.

    Parameters
    ----------
    delta
        Risk tolerance from the equilibrium portfolio
    weq
        weights of the assets in the equilibrium portfolio
    sigma
        Prior covariance matrix
    tau
        Coefficient of uncertainty in the prior estimate of the mean (pi)
    P
        Pick matrix for the view(s)
    Q
        Vector of view returns
    Omega
        Matrix of variance fo the views (diagonal)

    Returns
    -------
    tuple
        posteriorR, posteriorSigma, w, lmbda

    """
  # Reverse optimize and back out the equilibrium returns
  # This is formula (12) page 6.
    pi = weq.dot(sigma * delta)
  # We use tau * sigma many places so just compute it once
    ts = tau * sigma
  # Compute posterior estimate of the mean
  # This is a simplified version of formula (8) on page 4.
    middle = linalg.inv(np.dot(np.dot(P,ts),P.T) + Omega)
    er = np.expand_dims(pi,axis=0).T + np.dot(np.dot(np.dot(ts,P.T),middle),(Q - np.expand_dims(np.dot(P,pi.T),axis=1)))
  # Compute posterior estimate of the uncertainty in the mean
  # This is a simplified and combined version of formulas (9) and (15)
    posteriorSigma = sigma + ts - ts.dot(P.T).dot(middle).dot(P).dot(ts)
  # Compute posterior weights based on uncertainty in mean
    w = er.T.dot(linalg.inv(delta * posteriorSigma)).T
  # Compute lambda value
  # We solve for lambda from formula (17) page 7, rather than formula (18)
  # just because it is less to type, and we've already computed w*.
    lmbda = np.dot(linalg.pinv(P).T,(w.T * (1 + tau) - weq).T)
    return er, posteriorSigma, w, lmbda

def blacklitterman_nobayes(delta, weq, sigma, tau, P, Q, Omega) -> tuple:
    """
    This function performs the Black-Litterman blending of the prior and views into a new posterior estimate of the
    returns using the alternate reference model as shown in Idzorek's paper.
    See "A STEP-BY-STEP GUIDE TO THE BLACK-LITTERMAN MODEL, Incorporating user-specified confidence levels" by
    Thomas Idzorek".

    Parameters
    ----------
    delta
        Risk tolerance from the equilibrium portfolio
    weq
        weights of the assets in the equilibrium portfolio
    sigma
        Prior covariance matrix
    tau
        Coefficient of uncertainty in the prior estimate of the mean (pi)
    P
        Pick matrix for the view(s)
    Q
        Vector of view returns
    Omega
        Matrix of variance fo the views (diagonal)

    Returns
    -------
    tuple
        posteriorR, w
    """
    pi = weq.dot(sigma * delta)
    ts = tau * sigma
    middle = linalg.inv(np.dot(np.dot(P,ts),P.T) + Omega)
    er = np.expand_dims(pi,axis=0).T + np.dot(np.dot(np.dot(ts,P.T),middle),(Q - np.expand_dims(np.dot(P,pi.T),axis=1)))
    w = er.T.dot(linalg.inv(delta * sigma)).T
    # lmbda = np.dot(linalg.pinv(P).T, (w.T * (1 + tau) - weq).T)
    return er, w


def idz_omega(conf, P, Sigma) -> float:
    """
    This function computes the Black-Litterman parameters Omega from an Idzorek confidence.
    Parameters
    ----------
    conf
        Idzorek confidence specified as a decimal (50% as 0.5)
    P
        Pick matrix for the view
    Sigma
        Prior covariance matrix
    Returns
    -------
    omega
        Black-litterman uncertainty/confidence parameter
    """
    alpha = (1-conf) / conf
    omega = alpha * np.dot(np.dot(P, Sigma), P.T)
    return omega

if __name__ == '__main__':

    # # 1. test He & Litterman, 1999
    # weq = np.array([0.016, 0.022, 0.052, 0.055, 0.116, 0.124, 0.615])
    # C = np.array([[1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
    #               [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
    #               [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
    #               [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
    #               [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
    #               [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
    #               [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]])
    # Sigma = np.array([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187])
    # refPi = np.array([0.039, 0.069, 0.084, 0.090, 0.043, 0.068, 0.076])
    # assets = {'Australia', 'Canada   ', 'France   ', 'Germany  ', 'Japan    ', 'UK       ', 'USA      '}
    #
    # # Equilibrium covariance matrix
    # V = np.multiply(np.outer(Sigma,Sigma), C)
    #
    # delta = DefaultPara.delta
    # tau = DefaultPara.tau
    # tauV = DefaultPara.tau * V
    #
    # # Define view 1
    # # Germany will outperform the other European markets by 5%
    # # Market cap weight the P matrix
    # # Results should match Table 4, Page 21
    # P1 = np.array([0, 0, -.295, 1.00, 0, -.705, 0])
    # Q1 = np.array([0.05])
    # P = np.array([P1])
    # Q = np.array([Q1]);
    # Omega = np.dot(np.dot(P, tauV), P.T) * np.eye(Q.shape[0])
    # res = blacklitterman(delta, weq, V, tau, P, Q, Omega)
    # print(res)
    #
    # # Define view 2
    # # Canadian Equities will outperform US equities by 3%
    # # Market cap weight the P matrix
    # # Results should match Table 5, Page 22
    # P2 = np.array([0, 1.0, 0, 0, 0, 0, -1.0])
    # Q2 = np.array([0.03])
    # P = np.array([P1, P2])
    # Q = np.array([Q1, Q2])
    # Omega = np.dot(np.dot(P, tauV), P.T) * np.eye(Q.shape[0])
    # res = blacklitterman(delta, weq, V, tau, P, Q, Omega)
    # print(res)

    # 2. Take the values from Idzorek, 2005.
    weq = np.array([.193400, .261300, .120900, .120900, .013400, .013400, .241800, .034900])
    V = np.array([[.001005, .001328, -.000579, -.000675, .000121, .000128, -.000445, -.000437],
                  [.001328, .007277, -.001307, -.000610, -.002237, -.000989, .001442, -.001535],
                  [-.000579, -.001307, .059852, .027588, .063497, .023036, .032967, .048039],
                  [-.000675, -.000610, .027588, .029609, .026572, .021465, .020697, .029854],
                  [.000121, -.002237, .063497, .026572, .102488, .042744, .039443, .065994],
                  [.000128, -.000989, .023036, .021465, .042744, .032056, .019881, .032235],
                  [-.000445, .001442, .032967, .020697, .039443, .019881, .028355, .035064],
                  [-.000437, -.001535, .048039, .029854, .065994, .032235, .035064, .079958]])
    # refPi = np.array([0.0008,0.0067,0.0641,0.0408,0.0743,0.0370,0.0480,0.0660])
    assets = {'US Bonds  ', 'Intl Bonds', 'US Lg Grth', 'US Lg Value', 'US Sm Grth',
              'US Sm Value', 'Intl Dev Eq', 'Intl Emg Eq'}

    # Risk aversion of the market
    delta = DefaultPara.delta_idz

    # Coefficient of uncertainty in the prior estimate of the mean
    # from footnote (8) on page 11
    tau = DefaultPara.tau_idz
    tauV = tau * V

    # Define view 1
    # International Developed Equity will have an excess return of 5.25%
    # with a confidence of 25%.
    P1 = np.array([0, 0, 0, 0, 0, 0, 1, 0])
    Q1 = np.array([0.0525])
    conf1 = 0.25

    # Define view 2
    # International Bonds will outperform US Bonds by 0.0025 with a
    # confidence of 50%.
    P2 = np.array([-1, 1, 0, 0, 0, 0, 0, 0])
    Q2 = np.array([0.0025])
    conf2 = 0.50

    # Define View 3
    # US Large and Small Growth will outperform US Large and Small Value
    # by 0.02 with a confidence of 65%.
    P3 = np.array([0, 0, 0.90, -0.90, 0.10, -0.10, 0, 0])
    Q3 = np.array([0.02])
    conf3 = 0.65

    # Combine the views
    P = np.array([P1, P2, P3])
    Q = np.array([Q1, Q2, Q3])

    # Apply the views with simple Omega
    Omega = np.dot(np.dot(P, tauV), P.T)
    res = blacklitterman_nobayes(delta, weq, V, tau, P, Q, Omega)
    print(res)

    # Now apply the views using the Idzorek's method
    tauV = tau * V
    Omega = np.array(
        [[idz_omega(conf1, P1, tauV), 0, 0],
         [0, idz_omega(conf2, P2, tauV), 0],
         [0, 0, idz_omega(conf3, P3, tauV)]]
    )
    res = blacklitterman_nobayes(delta, weq, V, tau, P, Q, Omega)
    print(res)
