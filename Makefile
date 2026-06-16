# This makefile will help with running and cleaning the files

# Installing the requiredfiles
# Actually let's make the makefile handle everything

# we create an environment variable since Container can take decades loading
# Environment Name will be the same on every-computer since it will be uniue-Unless changed

environment_name = "mattp_pan_proj_env"

#  Create if It does not exist
# Activate - after running the project Deactivate It

target: requirements
