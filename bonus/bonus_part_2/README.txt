SETUP

This guide spells out how to install the package dependencies required
for running the GAN_demo.py script on CPU.

This particular script uses features (f-strings) from python 3.6. If you
have an older python version, either update your python version or replace
the f-strings with an alternative supported by your python version.

First, cd to the bonus_part_2 directory (the directory that contains this
README file and the GAN_demo.py script).

Then run the command "python -m venv bonus-part-2-env". This creates a
virtual environment by the name "bonus-part-2-env" in which the package
dependencies will be installed.

Next, run the appropriate activation command for your shell from the
list below:

POSIX

bash/zsh

$ source bonus-part-1-env/bin/activate

fish

$ source bonus-part-1-env/bin/activate.fish

csh/tcsh

$ source bonus-part-1-env/bin/activate.csh

PowerShell

$ bonus-part-1-env/bin/Activate.ps1


Windows

cmd.exe

C:\> bonus-part-1-env\Scripts\activate.bat

PowerShell

PS C:\> bonus-part-1-env\Scripts\Activate.ps1

You will notice that the command prompt changes to include the name
of the virtual environment. This indicates that any commands issued
to the shell will be executed in the context of the bonus-part-2-env
virtual environment.

Next, run the command "pip install -r requirements.txt".
This installs all of the package dependencies in the requirements.txt
file. These packages will only be accessible from within the context
of this virtual environment, and will not impact system-level
dependencies. In other words, your other projects are safe.

RUNNING THE SCRIPTS

Run the commmand "python GAN_demo.py" to train the Generative Adversarial
Network and to track its progress.

USAGE

To exit the virtual environment, run "deactivate".

To re-enter the virtual environment, run the appropriate activation
command from the list provided earlier.

You must be inside the virtual environment to run the scripts.
You will want to exit the virtual environment when you are done.

