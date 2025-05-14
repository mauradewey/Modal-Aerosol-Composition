
# likelihoods.py

# This file contains the likelihood functions for the CCN closure model.
# KnownSigmaGaussianLogLikelihood is a Guassian log-likelihood with sigma = 0.1 * observed data.
# GuassianLogLikelihood is a Gaussian log-likelihood with unknown sigma, which is estimated during optimization.

import pints
import numpy as np

# Define known sigma log-likelihood function
class KnownSigmaGaussianLogLikelihood(pints.LogPDF):

    def __init__(self, model, observed_data):
        """
        Custom Gaussian log-likelihood with known measurement noise (sigma).

        Argss:
        - model: A callable forward model.
        - observed_data: Observed dataset (numpy array).
        """
        
        self.model = model
        self.observed_data = np.array(observed_data)
        self._no = model.n_outputs() # Number of model outputs (CCN at 5 supersaturations)
        self._np = model.n_parameters() # total number of parameters (just model parameters)
        
        # for now we assume sigma is 10% of the observed data
        sigma = 0.1 * self.observed_data
        
        # Ensure all sigma values are positive floats
        if np.isscalar(sigma):
            sigma = np.ones(self._no) * float(sigma)  # Convert to array if scalar
        else:
            sigma = pints.vector(sigma)  # Ensure it's a numpy array
            if len(sigma) != self._no:
                raise ValueError("Sigma must have the same length as the number of outputs.")
        
        if np.any(sigma <= 0):
            raise ValueError("Sigma values must be positive.")
        
        # pre-calculate parts:
        self._offset = -0.5 * np.log(2 * np.pi)
        self._offset -= np.log(sigma)
        self._multip = -1 / (2.0 * sigma**2)
        
    def __call__(self, parameters):

        total_ccn = self.model(parameters) # Call the model with parameters

        # If model returns None (invalid parameters), reject sample
        if total_ccn is None:
            return -np.inf

        # Convert model output to numpy array
        total_ccn = np.array(total_ccn)

        # Ensure model output shape matches observed data
        if total_ccn.shape != self.observed_data.shape:
            return -np.inf

        # Compute residuals
        residuals = self.observed_data - total_ccn

        # Compute log-likelihood
        log_likelihood = np.sum(self._offset + self._multip * np.sum(residuals**2, axis=0))

        return log_likelihood

    def n_parameters(self):
        """Return the number of parameters expected."""
        return self._np
    

class GaussianLogLikelihood(pints.LogPDF):
    def __init__(self, model, observed_data):
        """
        Custom Gaussian log-likelihood with unknown measurement noise (sigma).

        Argss:
        - model: A callable forward model.
        - observed_data: Observed dataset (numpy array).
        """

        self.model = model
        self.observed_data = np.array(observed_data)
        self.n_model_parameters = model.n_parameters() # Number of model parameters
        self.n_total_parameters = model.n_parameters() + model.n_outputs()  # total number of parameters (model parameters + sigma)

    def __call__(self, parameters):
        """Compute Gaussian log-likelihood where sigma is also optimized."""
        model_parameters = parameters[:self.n_model_parameters]  # Model parameters
        sigma = np.array(parameters[self.n_model_parameters:])  # Last elements are sigma values
        
        # Ensure all sigma values are positive
        if np.any(sigma <= 0):
            return -np.inf  # Reject invalid sigmas

        total_ccn = self.model(model_parameters) # Call the model with parameters

        # If model returns None (invalid parameters), reject sample
        if total_ccn is None:
            return -np.inf

        # Convert model output to numpy array
        total_ccn = np.array(total_ccn)

        # Ensure model output shape matches observed data
        if total_ccn.shape != self.observed_data.shape:
            return -np.inf

        # Compute residuals
        residuals = self.observed_data - total_ccn

        # Compute log-likelihood
        log_likelihood = -0.5 * np.sum((residuals / sigma) ** 2) - np.sum(np.log(sigma)) - 0.5 * self.observed_data.size * np.log(2 * np.pi)

        return log_likelihood

    def n_parameters(self):
        """Return the number of parameters expected (including one sigma per output)."""
        return self.n_total_parameters