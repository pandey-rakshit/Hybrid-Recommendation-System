class SummaryFactory:
    def __init__(self):
        self.steps = {}

    def register_step(self, step_name, step_function):
        """Registers a summary step to the factory."""
        self.steps[step_name] = step_function

    def generate_summary(self, step_name, *args, **kwargs):
        """Generates the summary for the specified step."""
        if step_name not in self.steps:
            raise ValueError(f"Step '{step_name}' is not registered in the factory.")
        return self.steps[step_name](*args, **kwargs)
