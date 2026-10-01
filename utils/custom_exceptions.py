class OutOfBudgetError(Exception):
    """Exception raised for budget overflow

    Attributes:
        budget -- budget
        message -- explanation of the error
    """

    def __init__(self, budget, message="Budget is out of range"):
        self.budget = budget
        self.message = f"{message} : max = {self.budget}"
        super().__init__(self.message)