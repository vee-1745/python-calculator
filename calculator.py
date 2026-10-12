from __future__ import annotations

import ast
import keyword
import math
import operator
import re
import sys
from decimal import Decimal
from typing import Callable, Optional, Union

try:  # line editing and arrow-key history in the interactive prompt (Unix)
    import readline  # noqa: F401
except ImportError:
    pass

Number = Union[int, float]

MAX_EXPR_LEN = 500       # longest accepted expression
MAX_INT_BITS = 100_000   # largest integer result allowed (about 30,000 digits)
MAX_FACTORIAL = 1000
MAX_COMB = 10_000
MAX_HISTORY = 1000

BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

COMMANDS = {"help", "history", "vars", "clear", "mode", "q", "quit", "exit"}
QUIT_WORDS = {"q", "quit", "exit"}
ASSIGN_RE = re.compile(r"^([A-Za-z_]\w*)\s*=(?!=)\s*(.+)$")

HELP_TEXT = """\
Operators   + - * / // % **   and parentheses   (^ also means power)
Constants   pi  e  tau
Functions   sqrt abs round floor ceil exp ln log log2 sin cos tan asin acos atan
            sinh cosh tanh degrees radians factorial gcd lcm comb perm min max
            ln(x) is the natural log, log(x) is base 10, log(x, b) uses base b
Variables   x = 5 * 2      store a value, then use x in later expressions
            ans            the previous result
Commands    mode deg|rad   angle unit for sin/cos/tan/asin/acos/atan (now: {mode})
            history        show past calculations
            vars           show stored variables
            clear          forget variables and history
            help           show this message
            q / quit       exit

Examples    10 + 5 * 2        sqrt(2) ^ 2        log(8, 2)
            mode deg          sin(30)            comb(10, 3)
"""


class CalcError(Exception):
    """An error whose message is safe to show to the user."""


def fmt(value: Number) -> str:
    """Format a result for display (4.0 -> 4, 0.1+0.2 -> 0.3, huge ints -> 1.2e+5000)."""
    if isinstance(value, int):
        try:
            return str(value)
        except ValueError:  # beyond Python's int-to-str digit limit
            return format(Decimal(value), ".10e")
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.12g}"


class Calculator:
    def __init__(self) -> None:
        self.variables: dict[str, Number] = {}
        self.history: list[tuple[str, str]] = []
        self.degrees = False
        self.functions = self._build_functions()

    # ------------------------------------------------------------ properties
    @property
    def mode_name(self) -> str:
        return "deg" if self.degrees else "rad"

    def _reserved(self) -> set[str]:
        return set(CONSTANTS) | set(self.functions) | COMMANDS | {"ans"}

    # ------------------------------------------------------------- functions
    @staticmethod
    def _as_int(x: Number, name: str) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, float) and x.is_integer():
            return int(x)
        raise CalcError(f"{name}() needs whole-number arguments")

    def _build_functions(self) -> dict[str, Callable]:
        as_int = self._as_int

        def trig(fn: Callable[[float], float]) -> Callable[[Number], float]:
            def wrapped(x: Number) -> float:
                angle = math.radians(x) if self.degrees else x
                value = fn(angle)
                # snap float noise such as sin(pi) = 1.2e-16 to an exact 0
                return 0.0 if abs(value) < 1e-15 * max(1.0, abs(angle)) else value
            return wrapped

        def inverse_trig(fn: Callable[[float], float]) -> Callable[[Number], float]:
            def wrapped(x: Number) -> float:
                value = fn(x)
                return math.degrees(value) if self.degrees else value
            return wrapped

        def log(x: Number, base: Number = 10) -> float:
            if base == 10:
                return math.log10(x)
            if base == 2:
                return math.log2(x)
            return math.log(x, base)

        def factorial(n: Number) -> int:
            n = as_int(n, "factorial")
            if n < 0:
                raise CalcError("factorial() is not defined for negative numbers")
            if n > MAX_FACTORIAL:
                raise CalcError(f"factorial() argument too large (max {MAX_FACTORIAL})")
            return math.factorial(n)

        def gcd(*args: Number) -> int:
            return math.gcd(*[as_int(a, "gcd") for a in args])

        def lcm(*args: Number) -> int:
            return math.lcm(*[as_int(a, "lcm") for a in args])

        def comb(n: Number, k: Number) -> int:
            n, k = as_int(n, "comb"), as_int(k, "comb")
            if n > MAX_COMB:
                raise CalcError(f"comb() argument too large (max {MAX_COMB})")
            return math.comb(n, k)

        def perm(n: Number, k: Optional[Number] = None) -> int:
            n = as_int(n, "perm")
            if n > MAX_COMB:
                raise CalcError(f"perm() argument too large (max {MAX_COMB})")
            return math.perm(n) if k is None else math.perm(n, as_int(k, "perm"))

        return {
            "sqrt": math.sqrt, "abs": abs, "round": round,
            "floor": math.floor, "ceil": math.ceil,
            "exp": math.exp, "ln": math.log, "log": log, "log2": math.log2,
            "sin": trig(math.sin), "cos": trig(math.cos), "tan": trig(math.tan),
            "asin": inverse_trig(math.asin), "acos": inverse_trig(math.acos),
            "atan": inverse_trig(math.atan),
            "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
            "degrees": math.degrees, "radians": math.radians,
            "factorial": factorial, "gcd": gcd, "lcm": lcm,
            "comb": comb, "perm": perm, "min": min, "max": max,
        }

    # ------------------------------------------------------------ evaluation
    def evaluate(self, expr: str) -> Number:
        """Evaluate one expression safely and return a number."""
        expr = expr.strip().replace("^", "**")
        if not expr:
            raise CalcError("empty expression")
        if len(expr) > MAX_EXPR_LEN:
            raise CalcError(f"expression too long (max {MAX_EXPR_LEN} characters)")
        try:
            tree = ast.parse(expr, mode="eval")
            return self._eval(tree.body)
        except SyntaxError:
            raise CalcError("invalid syntax") from None
        except ZeroDivisionError:
            raise CalcError("division by zero") from None
        except OverflowError:
            raise CalcError("result too large") from None
        except RecursionError:
            raise CalcError("expression too complex") from None
        except ValueError as exc:
            raise CalcError(f"invalid input ({exc})") from None

    def _check(self, value: Number) -> Number:
        if isinstance(value, complex):
            raise CalcError("result is not a real number")
        if isinstance(value, float) and not math.isfinite(value):
            raise CalcError("result is not a finite number")
        if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
            raise CalcError("result too large")
        return value

    def _eval(self, node: ast.AST) -> Number:
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):  # excludes bool, complex, str
                return self._check(node.value)
            raise CalcError("only numbers are allowed")
        if isinstance(node, ast.Name):
            return self._lookup(node.id)
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
            return self._check(UNARY_OPS[type(node.op)](self._eval(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPS:
            left, right = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Pow):
                self._check_power(left, right)
            return self._check(BINARY_OPS[type(node.op)](left, right))
        if isinstance(node, ast.Call):
            return self._call(node)
        raise CalcError("unsupported syntax")

    @staticmethod
    def _check_power(base: Number, exponent: Number) -> None:
        """Refuse powers that would take forever or use huge amounts of memory."""
        if isinstance(base, int) and isinstance(exponent, int) and exponent > 0:
            if base.bit_length() * exponent > MAX_INT_BITS:
                raise CalcError("result too large")

    def _call(self, node: ast.Call) -> Number:
        if not isinstance(node.func, ast.Name) or node.func.id not in self.functions:
            name = node.func.id if isinstance(node.func, ast.Name) else "that expression"
            raise CalcError(f"unknown function: {name}")
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise CalcError("unsupported syntax")
        name = node.func.id
        args = [self._eval(a) for a in node.args]
        try:
            return self._check(self.functions[name](*args))
        except TypeError:
            raise CalcError(f"wrong number or type of arguments for {name}()") from None
        except ValueError as exc:
            raise CalcError(f"{name}(): {exc}") from None

    def _lookup(self, name: str) -> Number:
        if name in self.variables:
            return self.variables[name]
        if name in CONSTANTS:
            return CONSTANTS[name]
        if name == "ans":
            raise CalcError("no previous result yet")
        if name in self.functions:
            raise CalcError(f"'{name}' is a function; use {name}(...)")
        raise CalcError(f"unknown name '{name}'")

    # ------------------------------------------------------- line processing
    def _remember(self, text: str, value: Number) -> None:
        self.history.append((text, fmt(value)))
        del self.history[:-MAX_HISTORY]
        self.variables["ans"] = value

    def process(self, line: str) -> Optional[str]:
        """Handle one line of input and return the text to display (or None)."""
        line = line.strip()
        if not line:
            return None
        lower = line.lower()
        parts = lower.split()

        if lower == "help":
            return HELP_TEXT.format(mode=self.mode_name).rstrip()
        if lower == "history":
            if not self.history:
                return "No history yet."
            return "\n".join(f"{i:>3}. {expr}  ->  {res}"
                             for i, (expr, res) in enumerate(self.history, 1))
        if lower == "vars":
            if not self.variables:
                return "No variables defined."
            return "\n".join(f"{k} = {fmt(v)}" for k, v in self.variables.items())
        if lower == "clear":
            self.variables.clear()
            self.history.clear()
            return "Cleared variables and history."
        if parts[0] == "mode" and len(parts) <= 2 and "=" not in lower:
            if len(parts) == 2:
                if parts[1] in ("deg", "degrees"):
                    self.degrees = True
                elif parts[1] in ("rad", "radians"):
                    self.degrees = False
                else:
                    raise CalcError("usage: mode deg|rad")
            return f"Angle mode: {self.mode_name}"

        match = ASSIGN_RE.match(line)
        if match:
            name, expr = match.groups()
            if name in self._reserved() or keyword.iskeyword(name):
                raise CalcError(f"'{name}' is reserved and cannot be assigned")
            value = self.evaluate(expr)
            self.variables[name] = value
            self._remember(line, value)
            return f"{name} = {fmt(value)}"

        value = self.evaluate(line)
        self._remember(line, value)
        return f"= {fmt(value)}"


def repl(calc: Calculator) -> None:
    print("Advanced Python Calculator")
    print("Type an expression like 10 + 5 * 2, or 'help' to see everything it can do.")
    print("Type 'q' or 'quit' to exit.")
    while True:
        try:
            line = input(f"[{calc.mode_name}] >>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if line.strip().lower() in QUIT_WORDS:
            print("Goodbye!")
            return
        try:
            output = calc.process(line)
        except CalcError as exc:
            print(f"Error: {exc}")
            continue
        if output:
            print(output)


def main(argv: list[str]) -> int:
    calc = Calculator()
    if argv and argv[0] == "--deg":
        calc.degrees = True
        argv = argv[1:]
    if not argv:
        repl(calc)
        return 0
    try:
        print(fmt(calc.evaluate(" ".join(argv))))
    except CalcError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
