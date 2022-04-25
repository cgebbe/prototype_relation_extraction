from dataclasses import dataclass, asdict

@dataclass
class A:
    x: int
    y: float

a = A(2,0.1)

asdict(a)
a.__dict__


