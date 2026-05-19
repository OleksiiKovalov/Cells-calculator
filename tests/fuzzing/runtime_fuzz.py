"""Public runtime fuzzing facade and CLI entry point."""

from tests.fuzzing.config import *
from tests.fuzzing.generation import *
from tests.fuzzing.subprocess_runner import *
from tests.fuzzing.instrumentation import *
from tests.fuzzing.oracles import *
from tests.fuzzing.child_runner import *
from tests.fuzzing.signatures import *
from tests.fuzzing.minimizer import *
from tests.fuzzing.runner import main, parent_main, should_continue


if __name__ == "__main__":
    raise SystemExit(main())
