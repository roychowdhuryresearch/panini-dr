"""``python -m panini_dr.replay --mode {record,replay} --cassette DIR -- <runner args>``."""

import sys

from panini_dr.replay.launcher import main

sys.exit(main())
