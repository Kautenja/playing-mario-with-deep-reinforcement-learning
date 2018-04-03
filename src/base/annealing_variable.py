"""A wrapper for a variable that decays or grows over time."""


class AnnealingVariable(object):
    """A variable that decays or grows over time."""

    def __init__(self,
        initial_value: float,
        final_value: float,
        steps: int
    ) -> None:
        """
        Create a new annealing variable.

        Args:
            initial_value: the starting value for the variable
            final_value: the stopping value for the variable
            steps: the number of steps to get from start to stop

        Returns:
            None

        """
        # cast to expected (and necessary) types
        initial_value = float(initial_value)
        final_value = float(final_value)
        steps = int(steps)
        # assign instance members to self
        self.initial_value = initial_value
        self.final_value = final_value
        self.value = initial_value
        # get the geometric rate for the annealing
        self.rate = (final_value / initial_value)**(1.0 / steps)
        # determine whether to use max or min when bounding with the final
        # value. if the rate is above 1, then the value is growing
        if abs(self.rate) > 1:
            self.bound = min
        else:
            self.bound = max

    def step(self) -> None:
        """Perform a step to decay or grow the variable."""
        self.value = self.bound(self.value * self.rate, self.final_value)


# explicitly define the outward facing API of this module
__all__ = ['AnnealingVariable']
