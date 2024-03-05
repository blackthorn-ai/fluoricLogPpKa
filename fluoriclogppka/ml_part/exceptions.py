class FeatureNotFoundError(Exception):
    """
    Exception raised when a requested feature is not found.

    Attributes:
        feature_name (str): The name of the feature that was not found.
    """
    def __init__(self, feature_name):
        self.feature_name = feature_name
        super().__init__(f"Feature name: {feature_name} is Not Found")


class InvalidMoleculeTypeError(Exception):
    """
    Exception raised when attempting to calculate a feature in the molecule 
    where it is not possible due to the impossibility of calculating X1, X2, R1 or R2.

    Attributes:
        X1 (int): Atom id in cycle, that connects NH2 or COOH to the molecule.
        X2 (int): Atom id in cycle, that connects fluorine functional group to the molecule.
        R1 (int): NH2 or COOH functional group center atom id.
        R2 (int): fluorine functional group center atom id.
        feature_name (str): The name of the feature.
    """
    def __init__(self, X1, X2, R1, R2, feature_name):
        self.X1 = X1
        self.X2 = X2
        self.R1 = R1
        self.R2 = R2
        self.feature_name = feature_name
        super().__init__(f"Cannot calculate: {feature_name} due to invalid values:\n\
                         X1: {X1}\n\
                         X2: {X2}\n\
                         R1: {R1}\n\
                         R2: {R2}")
