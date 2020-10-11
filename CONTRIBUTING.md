# How to Contribute

Contributions and patches are always welcomed, following are some few things before submitting your contriibution.

## Format

Format your code in following way:
Use Black formatter 

```
black -l 79 file_name.py
```

## Tests

Complete all tests in tests folder by using following code:

```
ls tests/*.py|xargs -n 1 -P 3 python &> test.log &&echo "All passed" || echo "Failed! Search keyword FAILED in test.log"

```

