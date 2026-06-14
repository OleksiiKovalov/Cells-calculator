"""Public runtime fuzzing facade and CLI entry point."""

from tests_local.fuzzing.config import *
from tests_local.fuzzing.generation import *
from tests_local.fuzzing.subprocess_runner import *
from tests_local.fuzzing.instrumentation import *
from tests_local.fuzzing.oracles import *
from tests_local.fuzzing.child_runner import *
from tests_local.fuzzing.signatures import *
from tests_local.fuzzing.minimizer import *
from tests_local.fuzzing.runner import main, parent_main, should_continue


if __name__ == "__main__":
    raise SystemExit(main())
