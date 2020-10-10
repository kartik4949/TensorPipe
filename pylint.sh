echo "Checking File : $1"
python -m pylint --rcfile=.pylintrc  --disable=deprecated-module --const-rgx='[a-z_][a-z0-9_]{2,30}$' $1
