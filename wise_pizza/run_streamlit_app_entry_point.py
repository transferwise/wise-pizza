import os
import runpy
import sys

import wise_pizza


def main() -> None:
    streamlit_script_path = os.path.join(os.path.dirname(wise_pizza.__file__), "streamlit_app.py")
    sys.argv = ["streamlit", "run", streamlit_script_path ]
    runpy.run_module("streamlit", run_name="__main__")


if __name__ == "__main__":
    main()
