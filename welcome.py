import autofit as af
from autoconf import check_version

check_version(af.__version__)

input(
    "########################################\n"
    "### WELCOME TO HOWTOFIT               ###\n"
    "########################################\n\n"
    "This script runs a few checks to ensure PyAutoFit is set up correctly.\n"
    ""
    "Once they pass, read through start_here.py (or start_here.ipynb) and then open\n"
    "scripts/chapter_1_introduction/start_here.py to begin the tutorial series.\n\n"
    "\n"
    "########################################\n"
    "### HOWTOFIT WORKING DIRECTORY         ###\n"
    "########################################\n\n"
    """
    PyAutoFit assumes that the `HowToFit` directory is the Python working directory.
    This means that, when you run a tutorial script, you should run it from the `HowToFit`
    repository root as follows:


    cd path/to/HowToFit (if you are not already in HowToFit).
    python3 scripts/chapter_1_introduction/tutorial_1_models.py


    The reasons for this are so that PyAutoFit can:

    - Load configuration settings from config files in the `HowToFit/config` folder.
    - Write simulated tutorial data to `HowToFit/dataset/`.
    - Output the results of model fits to `HowToFit/output/`.

    If you get errors relating to importing modules, loading data or writing output, it is
    most likely because you are not running the script with `HowToFit` as the working
    directory.

    [Press Enter to continue]"""
)

input(
    "########################################\n"
    "### MATPLOTLIB BACKEND                 ###\n"
    "########################################\n\n"
    """
    Figures produced by the tutorials are rendered with `matplotlib`. Depending on your
    system, you may need to configure the matplotlib backend.

    [Press Enter to finish]"""
)
