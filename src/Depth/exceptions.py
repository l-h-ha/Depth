class BaseError(Exception):
    pass

###
###

class ComputationError(BaseError):
    pass

class BatchCalculationError(ComputationError):
    pass

class GradientComputeError(ComputationError):
    pass

###
###

class ComponentNotBuiltError(BaseError):
    pass