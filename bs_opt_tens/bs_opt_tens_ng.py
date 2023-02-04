try:
    import nevergrad as ng
    with_ng = True
except Exception as e:
    with_ng = False


from .bs_opt_tens import BsOptTens


class BsOptTensNg(BsOptTens):
    def __init__(self, name='ng'):
        # Abstract class
        super().__init__(name)
        self.optimizer_class = None

    def _init(self):
        if not with_ng:
            self.err = 'Need "nevergrad" module'
            return

    def _optimize(self):
        optimizer_ = eval(f'ng.optimizers.{self.optimizer_class}')
        optimizer = optimizer_(
            parametrization=ng.p.TransitionChoice(range(self.n[0]),
            repetitions=len(self.n)),
            budget=self.M,
            num_workers=1)

        recommendation = optimizer.provide_recommendation()

        for _ in range(optimizer.budget):
            x = optimizer.ask()
            optimizer.tell(x, self.f(x.value))

        self.i_opt = optimizer.provide_recommendation().value
        self.y_opt = self.f(self.i_opt)
