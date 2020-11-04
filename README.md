# Introduction
Functions to run regression in Python.

# Installation
<!---INSTALLATION_STANDARD_START.-->
I found standard methods for managing submodules to be a little complicated so I use my own method for managing my submodules. I use the mysubmodules project to quickly install these. To install the project, it's therefore sensible to download the mysubmodules project and then use a script in the mysubmodules project to install the submodules for this project.

If you are in the directory where you wish to download python-regrun i.e. if you wish to install the project at /home/files/python-regrun/ and you are at /home/files/, and you have not already cloned the directory to /home/files/python-regrun/, you can run the following commands to download the directory:

```
git clone https://github.com/c-d-cotton/mysubmodules.git getmysubmodules
python3 getmysubmodules/singlegitmodule.py python-regrun --downloadmodule --deletegetsubmodules
```

The option --downloadmodule downloads the actual module before installing the submodules. The option --deletegetsubmodules deletes the getsubmodules project after the submodules are installed.

If you have already downloaded projectdir to the folder /home/files/python-regrun/, you can add the submodules by running the following commands from the directory /home/files/:
```
git clone https://github.com/c-d-cotton/mysubmodules.git getmysubmodules
python3 getmysubmodules/singlegitmodule.py python-regrun --deletegetsubmodules
```
<!---INSTALLATION_STANDARD_END.-->



# Basic steps for running multiple similar regressions.
1. Generate matrices for regressions.
2. Run regressions.
3. Save/append basic results of regressions in tex/txt form.
4. Construct tabular summary of many regressions.
5. Take tabular and construct table summary.

I also have a simple function that does all of this in one go for OLS.
